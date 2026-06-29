#ifndef ONNX_ALQUIMIA_INTERFACE_H
#define ONNX_ALQUIMIA_INTERFACE_H

#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_containers.h"
#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
#if ALQUIMIA_HAVE_ONNX
    void lsurf_alquimia_setup(
        void **engine_internal_state,
        bool hands_off,
        AlquimiaSizes *sizes,
        AlquimiaEngineFunctionality *functionality,
        AlquimiaEngineStatus *status);

    void lsurf_alquimia_shutdown(
        void *engine_internal_state,
        AlquimiaEngineStatus *status);

    void lsurf_alquimia_processcondition(
        void *engine_internal_state,
        AlquimiaGeochemicalCondition *condition,
        AlquimiaProperties *properties,
        AlquimiaState *state,
        AlquimiaAuxiliaryData *aux_data,
        AlquimiaEngineStatus *status);

    void lsurf_alquimia_reactionstepoperatorsplit(
        void *engine_internal_state,
        double delta_t,
        AlquimiaProperties *properties,
        AlquimiaState *state,
        AlquimiaAuxiliaryData *aux_data,
        AlquimiaEngineStatus *status);

    void lsurf_alquimia_getauxiliaryoutput(
        void *engine_internal_state,
        AlquimiaProperties *properties,
        AlquimiaState *state,
        AlquimiaAuxiliaryData *aux_data,
        AlquimiaAuxiliaryOutputData *aux_output,
        AlquimiaEngineStatus *status);

    void lsurf_alquimia_getproblemmetadata(
        void *engine_internal_state,
        AlquimiaProblemMetaData *meta_data,
        AlquimiaEngineStatus *status);
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* ALQUIMIA_ONNX_INTERFACE_H_ */