"""Exercise MCP discovery, session ownership, and failures without native engines."""

import json
from pathlib import Path
import sys
import threading
import time

import anyio
from mcp import Client
import pytest

from mcp_alquimia.mcp_server import create_server
from mcp_alquimia.session_manager import SessionManager
from mcp_alquimia.subprocess_client import EngineProcessError


@pytest.fixture
def anyio_backend():
    """Use the asyncio backend supported by the project's installed MCP SDK."""
    return "asyncio"


class FakeEngineProcess:
    """Small stateful stand-in used to probe ownership and concurrent requests."""

    instances = []
    fail_operation = None

    def __init__(self, executable, run_dir, timeout):
        self.run_dir = run_dir
        self.closed = False
        self.time = 0
        self.steps = 0
        self.instances.append(self)

    def request(self, operation, **parameters):
        """Return a snapshot or a deliberate engine failure."""
        if self.fail_operation == operation:
            raise EngineProcessError(f"{operation} failed; log: {self.run_dir}/engine_output.log")
        if operation == "setup":
            return {"primary_species": ["H"], "minerals": []}
        if operation == "react":
            previous = self.time
            time.sleep(0.01)
            self.time = previous + parameters["timestep"]
            self.steps += 1
        return {"time": self.time, "max_steps": self.steps, "total_mobile": [1.0]}

    def close(self):
        """Record that this process was reaped."""
        self.closed = True


@pytest.fixture
def manager(monkeypatch, repository_root, tmp_path):
    """Provide an isolated session owner using fake native processes."""
    monkeypatch.setattr("mcp_alquimia.session_manager.AlquimiaEngineProcess", FakeEngineProcess)
    monkeypatch.setattr(FakeEngineProcess, "instances", [])
    monkeypatch.setattr(FakeEngineProcess, "fail_operation", None)
    return SessionManager(
        repository_root,
        Path(sys.executable),
        tmp_path / "runs",
        max_sessions=2
        )


async def call(client, name, **arguments):
    """Call a tool and require a structured successful result."""
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result
    assert isinstance(result.structured_content, dict)
    return result.structured_content


@pytest.mark.anyio
async def test_discovery_and_missing_runtime(repository_root, tmp_path):
    """Discovery needs no compiled runtime; setup returns an actionable error."""
    manager = SessionManager(repository_root, tmp_path / "missing", tmp_path / "runs")
    async with Client(create_server(manager)) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}
        assert set(tools) == {"list_cases", "setup", "initialize", "react", "get_results", "close"}
        assert tools["setup"].input_schema["properties"]["case"]["enum"] == [
            "calcite-pflotran", "calcite-crunchflow", "ex8-nn-batch1-onnx"]
        assert tools["react"].input_schema["properties"]["dt"]["exclusiveMinimum"] == 0
        assert len((await call(client, "list_cases"))["cases"]) == 3
        error = await client.call_tool("setup", {"case": "calcite-pflotran"})
        assert error.is_error
        assert "Build it explicitly" in error.content[0].text
        assert not (tmp_path / "runs").exists()


@pytest.mark.anyio
async def test_sessions_lifecycle_isolation_and_disconnect(manager):
    """Separate sessions retain separate states; disconnect reaps open processes."""
    async with Client(create_server(manager)) as client:
        first = await call(client, "setup", case="calcite-pflotran")
        second = await call(client, "setup", case="ex8-nn-batch1-onnx")
        a, b = first["session_id"], second["session_id"]
        assert a != b
        limited = await client.call_tool("setup", {"case": "calcite-crunchflow"})
        assert limited.is_error
        early = await client.call_tool("react", {"session_id": a, "dt": 5})
        assert early.is_error
        assert "Call initialize" in early.content[0].text
        await call(client, "initialize", session_id=a)
        await call(client, "initialize", session_id=b)
        repeated = await client.call_tool("initialize", {"session_id": a})
        assert repeated.is_error
        # Simultaneous MCP calls must not interleave writes to the same process.
        async with anyio.create_task_group() as group:
            async def react(dt):
                await call(client, "react", session_id=a, dt=dt)
            group.start_soon(react, 2.0)
            group.start_soon(react, 3.0)
        state = (await call(client, "get_results", session_id=a))["state"]
        assert (state["time"], state["max_steps"]) == (5, 2)
        assert (await call(client, "get_results", session_id=b))["state"]["time"] == 0
        assert (await call(client, "close", session_id=a))["closed"] is True
        assert (await call(client, "close", session_id=a))["closed"] is False
        unknown = await client.call_tool("get_results", {"session_id": a})
        assert unknown.is_error
        # Closing one session releases its slot.
        await call(client, "setup", case="calcite-crunchflow")
    assert all(process.closed for process in FakeEngineProcess.instances)
    assert manager._sessions == {}
    saved = json.loads((Path(first["run_dir"]) / "simulation_results.json").read_text())
    assert saved["status"] == "closed"
    assert [row["time"] for row in saved["history"]][-1] == 5
    assert len(saved["history"]) == 3


@pytest.mark.anyio
async def test_invalid_arguments_preserve_session(manager):
    """Schema errors and invalid dt never reach or invalidate the engine."""
    async with Client(create_server(manager)) as client:
        bad_case = await client.call_tool("setup", {"case": "arbitrary-file"})
        assert bad_case.is_error
        assert not manager.run_root.exists()
        setup = await call(client, "setup", case="calcite-pflotran")
        session_id = setup["session_id"]
        await call(client, "initialize", session_id=session_id)
        for dt in (-1, 0, True, "5"):
            # Call from the client
            result = await client.call_tool("react", {"session_id": session_id, "dt": dt})
            assert result.is_error
        for dt in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                # Directly call the react function
                manager.react(session_id, dt)
        assert (await call(client, "get_results", session_id=session_id))["state"]["time"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["setup", "initialize", "react"])
async def test_native_failure_releases_session_and_keeps_diagnosis(manager, monkeypatch, operation):
    """Failed operations release their slot and preserve engine diagnostics."""
    async with Client(create_server(manager)) as client:
        if operation == "setup":
            arguments = {"case": "calcite-pflotran"}
        else:
            setup = await call(client, "setup", case="calcite-pflotran")
            arguments = {"session_id": setup["session_id"]}
            if operation == "react":
                await call(client, "initialize", **arguments)
                arguments["dt"] = 5
        monkeypatch.setattr(FakeEngineProcess, "fail_operation", operation)
        result = await client.call_tool(operation, arguments)
        assert result.is_error
        assert f"{operation} failed" in result.content[0].text
        assert "engine_output.log" in result.content[0].text
        assert all(process.closed for process in FakeEngineProcess.instances)
        assert not manager._sessions
        failures = list(manager.run_root.glob("*/simulation_failure.json"))
        assert len(failures) == 1
        assert json.loads(failures[0].read_text())["operation"] == operation


@pytest.mark.anyio
async def test_cleanup_continues_after_one_close_failure(manager, monkeypatch):
    """A shutdown error must not leave other sessions running."""
    async with Client(create_server(manager)) as client:
        await call(client, "setup", case="calcite-pflotran")
        await call(client, "setup", case="calcite-crunchflow")
        process = FakeEngineProcess.instances[0]
        def fail_close():
            process.closed = True
            raise EngineProcessError("shutdown failed")
        monkeypatch.setattr(process, "close", fail_close)
    assert all(process.closed for process in FakeEngineProcess.instances)
    assert not manager._sessions


@pytest.mark.anyio
async def test_cancelled_setup_does_not_orphan_process(manager, monkeypatch):
    """An in-flight setup remains owned when its client cancels and disconnects."""
    
    # Flags to pause and resume the background thread
    started = threading.Event()
    release = threading.Event()
    request = FakeEngineProcess.request # Backup the original normal execution

    # Intercept the 'setup' operation to freeze it mid-flight
    def delayed_request(self, operation, **parameters):
        if operation == "setup":
            started.set()   # 1. Notify main thread that setup has begun
            assert release.wait(timeout=5)  # 2. Freeze here until main thread releases it
        return request(self, operation, **parameters)

    monkeypatch.setattr(FakeEngineProcess, "request", delayed_request)
    async with Client(create_server(manager)) as client:
        async with anyio.create_task_group() as group:
            async def setup():
                await client.call_tool("setup", {"case": "calcite-pflotran"})
            
            # Start the setup request in the background (another thread)
            # main thread continues
            group.start_soon(setup)
            
            # Wait until the background thread gets stuck at our freeze point
            with anyio.fail_after(5):
                while not started.is_set():
                    await anyio.sleep(0.001)
                    
            # Simulate a client disconnect by cancelling the task mid-flight
            group.cancel_scope.cancel()
            
            # Unfreeze the thread so the system can handle the cancellation and clean up
            release.set()
    # Verify the system cleaned up properly
    assert len(FakeEngineProcess.instances) == 1
    assert FakeEngineProcess.instances[0].closed
    assert not manager._sessions
