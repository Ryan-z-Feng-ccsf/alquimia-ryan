#ifndef ONNX_ALQUIMIA_INTERFACE_H
#define ONNX_ALQUIMIA_INTERFACE_H

#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_containers.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
#if ALQUIMIA_HAVE_ONNX
    void onnx_alquimia_setup(
        const char* input_filename,
        bool hands_off,
        void* onnx_engine_state,
        AlquimiaSizes* sizes,
        AlquimiaEngineFunctionality* functionality,
        AlquimiaEngineStatus* status);

    void onnx_alquimia_shutdown(
        void* onnx_engine_state,
        AlquimiaEngineStatus* status);

    void onnx_alquimia_processcondition(
        void* onnx_engine_state,
        AlquimiaGeochemicalCondition* condition,
        AlquimiaProperties* props,
        AlquimiaState* state,
        AlquimiaAuxiliaryData* aux_data,
        AlquimiaEngineStatus* status);

    void onnx_alquimia_reactionstepoperatorsplit(
        void* onnx_engine_state,
        double delta_t,
        AlquimiaProperties* props,
        AlquimiaState* state,
        AlquimiaAuxiliaryData* aux_data,
        int natural_id,
        AlquimiaEngineStatus* status);

    void onnx_alquimia_getauxiliaryoutput(
        void* onnx_engine_state,
        AlquimiaProperties* props,
        AlquimiaState* state,
        AlquimiaAuxiliaryData* aux_data,
        AlquimiaAuxiliaryOutputData* aux_out,
        AlquimiaEngineStatus* status);

    void onnx_alquimia_getproblemmetadata(
        void* onnx_engine_state,
        AlquimiaProblemMetaData* meta_data,
        AlquimiaEngineStatus* status);
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* ONNX_ALQUIMIA_INTERFACE_H_ */