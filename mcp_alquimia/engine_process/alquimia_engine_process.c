/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

/*
** Alquimia Copyright (c) 2013-2016, The Regents of the University of California,
** through Lawrence Berkeley National Laboratory (subject to receipt of any
** required approvals from the U.S. Dept. of Energy).  All rights reserved.
**
** Alquimia is available under a BSD license. See LICENSE.txt for more
** information.
**
** If you have questions about your rights to use or distribute this software,
** please contact Berkeley Lab's Technology Transfer and Intellectual Property
** Management at TTD@lbl.gov referring to Alquimia (LBNL Ref. 2013-119).
**
** NOTICE.  This software was developed under funding from the U.S. Department
** of Energy.  As such, the U.S. Government has been granted for itself and
** others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide
** license in the Software to reproduce, prepare derivative works, and perform
** publicly and display publicly.  Beginning five (5) years after the date
** permission to assert copyright is obtained from the U.S. Department of Energy,
** and subject to any subsequent five (5) year renewals, the U.S. Government is
** granted for itself and others acting on its behalf a paid-up, nonexclusive,
** irrevocable, worldwide license in the Software to reproduce, prepare derivative
** works, distribute copies to the public, perform publicly and display publicly,
** and to permit others to do so.
*/

/* **************************************************************************** 
**
** MCP Alquimia simulation wrapper
**
** Authors: 
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * Persistent Alquimia engine process. Protocol replies use a dedicated file descriptor (fd);
**  * engine stdout/stderr belong to the caller's log, never the protocol stream. 
**  @file alquimia_engine_process.c
**  @brief IPC protocol and execution lifecycle for the computational engine.
**
**  Communication Protocol Specification:
**  - Architecture: Line-delimited JSON (NDJSON) over standard stream channels.
**  - Inbound Stream (stdin):
**     * Requests are read line-by-line via fgets() with an upper bound buffer (e.g., 16 KB).
**     * Each request payload MUST be a single-line, unformatted JSON object strictly
**       terminated by a newline ('\n'). Multi-line JSON formatting within a single request
**       is prohibited to prevent stream fragmentation and parsing deadlocks.
**  - Outbound Stream (response-fd):
**     * Responses are emitted via a dedicated file descriptor encapsulated by fdopen().
**     * Structured replies are serialized using cJSON_PrintUnformatted() to ensure compact,
**       single-line emission followed by an explicit trailing newline ('\n').
**     * Each frame execution forces an immediate fflush() to bypass C runtime buffering
**       and deliver prompt notifications to the parent process.
**  - Error Handling & Wire Invariants:
**     * Payloads lacking a terminating newline within the line buffer capacity trigger
**       an immediate request error and cycle termination.
**     * Enforces case-sensitive key lookups and strict null-terminated JSON validations.
**
** **************************************************************************** 
*/

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_memory.h"
#include "alquimia_engine_process_helpers.h"
#if ALQUIMIA_NEED_PETSC
#include "petsc.h"
#endif

int main(int argc, char** argv)
{
  // Open a different pipeline other than stdin, stdout, stderr
  if (argc != 3 || strcmp(argv[1], "--response-fd") != 0)
  {
    fprintf(stderr, "Usage: alquimia_engine_process --response-fd FD\n");
    return EXIT_FAILURE;
  }

  char* end;
  errno = 0;
  long fd = strtol(argv[2], &end, 10);
  if (errno || *end || fd < 3 || fd > 1024 * 1024)
    return EXIT_FAILURE;

  FILE* response = fdopen((int)fd, "w");
  if (!response)
    return EXIT_FAILURE;
  int exit_code = EXIT_SUCCESS;

// Initialize
#if ALQUIMIA_NEED_PETSC
  // Ignore CLI command 
  PetscErrorCode petsc_error = PetscInitializeNoArguments();
  if (petsc_error)
  {
    Send(response, Error("startup", (int)petsc_error, "PETSc initialization failed"));
    fclose(response);
    return EXIT_FAILURE;
  }
  PetscInitializeFortran();
#endif

  // Initialize
  SimulationEngineSession simulation_session = {0};
  AllocateAlquimiaEngineStatus(&simulation_session.status);

  bool closing = false;

  // Parse the JSON request from Python Process
  char line[16384];

  if (!Send(response, Success(cJSON_CreateString("ready")))) 
    exit_code = EXIT_FAILURE;

  // Read the input from the Python process
  while (!exit_code && fgets(line, sizeof(line), stdin)) 
  {
    // Ensure complete JSON
    if (!strchr(line, '\n')) {
      Send(response, Error("request", -1, "Request is too long or not newline terminated"));
      exit_code = EXIT_FAILURE;
      break;
    }

    // Parse JSON to cJSON object
    cJSON* request = cJSON_ParseWithOpts(line, NULL, true);

    const char* operation = JsonToString(request, "operation");

    // Closing
    if (operation && strcmp(operation, "close") == 0) 
    {
      closing = true;
      cJSON_Delete(request);
      break;
    }

    // Initialize the engine or initialize the process condition or Run the simulation
    cJSON* reply = cJSON_IsObject(request) ? Execute(&simulation_session, request) :
        Error("request", -1, "Expected a JSON object");
    cJSON_Delete(request);

    // Send the simulation results
    if (!Send(response, reply))
      exit_code = EXIT_FAILURE;
  }

  // Clean up
  int shutdown_error = Cleanup(&simulation_session);
  if (shutdown_error) 
    exit_code = EXIT_FAILURE;
  FreeAlquimiaEngineStatus(&simulation_session.status);

#if ALQUIMIA_NEED_PETSC
  if (PetscFinalize())
    exit_code = EXIT_FAILURE;
#endif

  if (closing && !Send(response, exit_code ?
      Error("close", -1, "Native shutdown/finalization failed; see engine log") :
      Success(cJSON_CreateString("closed")))) 
    exit_code = EXIT_FAILURE;
  fclose(response);
  return exit_code;
}
