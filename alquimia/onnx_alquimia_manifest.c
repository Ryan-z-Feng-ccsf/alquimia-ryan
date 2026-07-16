/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

#include "alquimia/onnx_alquimia_manifest.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"

/* cJSON is confined to this translation unit. The interface layer receives
** only Alquimia-owned structures whose strings remain valid after the parse
** tree is deleted. */

/**
 * @brief Copies a fixed error detail when the caller supplied a buffer.
 * @param message Destination buffer, which may be NULL.
 * @param size Size of @p message in bytes.
 * @param detail Null-terminated error detail.
 */
static void SetError(char *message, size_t size, const char *detail)
{
  if (message != NULL && size > 0)
  {
    snprintf(message, size, "%s", detail);
  }
}

/**
 * @brief Creates a C-allocator-owned copy of a parsed JSON string.
 * @param value Null-terminated source string owned by cJSON.
 * @return Newly allocated storage, or NULL on allocation failure.
 */
static char *CopyString(const char *value)
{
  size_t length = strlen(value);
  char *copy = (char *)malloc(length + 1);
  if (copy != NULL)
  {
    memcpy(copy, value, length + 1);
  }
  return copy;
}

/**
 * @brief Checks a property name against an exact, case-sensitive allowlist.
 * @param name Property name to check.
 * @param allowed Accepted property names.
 * @param num_allowed Number of entries in @p allowed.
 * @return True only for an exact allowlist match.
 */
static bool IsAllowedProperty(
    const char *name,
    const char *const *allowed,
    size_t num_allowed)
{
  size_t i;
  for (i = 0; i < num_allowed; ++i)
  {
    if (strcmp(name, allowed[i]) == 0)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief Rejects unknown and duplicate members in one manifest object.
 * @param object cJSON object whose direct children are inspected.
 * @param allowed Exact property allowlist for this object type.
 * @param num_allowed Number of entries in @p allowed.
 * @param context Human-readable object name used in errors.
 * @param error_message Destination for the first validation error.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every member is allowed and appears at most once.
 *
 * cJSON permits duplicate object keys and lookup returns only one occurrence,
 * so strict validation must traverse the complete child list explicitly.
 */
static bool ValidateProperties(
    const cJSON *object,
    const char *const *allowed,
    size_t num_allowed,
    const char *context,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *property;
  size_t i;

  /* Triple loop to check duplicate keys */
  /* Trade-off between space and time performance */
  /* Right here, pick the space */
  /* Uses an O(N^2 * M) triple loop to detect duplicate keys without allocating 
  ** extra memory, prioritizing space efficiency over time. */
  cJSON_ArrayForEach(property, object)
  {
    /* Check if the key is empty */
    /* Check if the key is valid */
    if (property->string == NULL ||
        !IsAllowedProperty(property->string, allowed, num_allowed))
    {
      snprintf(error_message, error_message_size,
               "Unknown property '%s' in %s.",
               property->string == NULL ? "" : property->string, context);
      return false;
    }
    /* Second layer of the loop, traverse the allowed keys */
    for (i = 0; i < num_allowed; ++i)
    {
      /* If the key was found */
      if (strcmp(property->string, allowed[i]) == 0)
      {
        /* A new cJSON used to find the duplicate key */
        const cJSON *other;
        /* Record the number of each key */
        int count = 0;
        /* Traverse the key from the start to record the number of each key */
        cJSON_ArrayForEach(other, object)
        {
          if (other->string != NULL &&
              strcmp(other->string, allowed[i]) == 0)
          {
            ++count;
          }
        }
        /* If the count > 1, duplicate key */
        if (count > 1)
        {
          snprintf(error_message, error_message_size,
                   "Duplicate property '%s' in %s.", allowed[i], context);
          return false;
        }
        break;
      }
    }
  }
  return true;
}

/**
 * @brief Reads and copies a required nonempty string property.
 * @param object Object containing the property.
 * @param name Exact property name.
 * @param context Human-readable object name used in errors.
 * @param value Receives C-allocator-owned storage on success.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when the property is a nonempty string and copying succeeds.
 *
 * Copying detaches the manifest representation from the cJSON tree lifetime.
 */
static bool GetRequiredString(
    const cJSON *object,
    const char *name,
    const char *context,
    char **value,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *item = cJSON_GetObjectItemCaseSensitive(object, name);
  if (!cJSON_IsString(item) || item->valuestring == NULL ||
      item->valuestring[0] == '\0')
  {
    snprintf(error_message, error_message_size,
             "Required property '%s' in %s must be a nonempty string.",
             name, context);
    return false;
  }
  *value = CopyString(item->valuestring);
  if (*value == NULL)
  {
    SetError(error_message, error_message_size,
             "Memory allocation failed while loading ONNX manifest.");
    return false;
  }
  return true;
}

/**
 * @brief Reads a required integer representable as a nonnegative C int.
 * @param object Object containing the property.
 * @param name Exact property name.
 * @param context Human-readable object name used in errors.
 * @param value Receives the parsed integer on success.
 * @param error_message Destination for validation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when the JSON number is integral and lies in [0, INT_MAX].
 *
 * cJSON stores numbers as double, so the round-trip comparison rejects
 * fractional values before conversion to the manifest's integer fields.
 */
static bool GetRequiredInteger(
    const cJSON *object,
    const char *name,
    const char *context,
    int *value,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *item = cJSON_GetObjectItemCaseSensitive(object, name);
  if (!cJSON_IsNumber(item) || item->valuedouble < 0.0 ||
      item->valuedouble > INT_MAX ||
      (double)(int)item->valuedouble != item->valuedouble)
  {
    snprintf(error_message, error_message_size,
             "Required property '%s' in %s must be a nonnegative integer.",
             name, context);
    return false;
  }
  *value = (int)item->valuedouble;
  return true;
}

/**
 * @brief Resolves the manifest model entry to an owned filesystem path.
 * @param manifest_path Path used to open the manifest.
 * @param model Nonempty model entry from the manifest.
 * @param model_path Receives newly allocated path storage.
 * @param error_message Destination for overflow or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when path construction succeeds.
 *
 * Absolute paths are preserved. Relative paths use the manifest's directory,
 * not the process working directory; existence is checked later by setup.
 */
static bool ResolveModelPath(
    const char *manifest_path,
    const char *model,
    char **model_path,
    char *error_message,
    size_t error_message_size)
{
  /* A pointer points to the last '/' */
  const char *separator;
  size_t directory_length;
  size_t model_length = strlen(model);

  /* Absolute path /usr/... */
  if (model[0] == '/')
  {
    *model_path = CopyString(model);
  }
  /* relative path models/.onnx */
  else
  {
    /* Find the last '/' */
    separator = strrchr(manifest_path, '/');
    /* Get the length of the directory inclding the last '/' */
    directory_length = separator == NULL ? 0 : (size_t)(separator - manifest_path) + 1;
    /* Check if the path exceeds the size_t */
    if (directory_length > SIZE_MAX - model_length - 1)
    {
      SetError(error_message, error_message_size,
               "Resolved ONNX model path is too long.");
      return false;
    }
    *model_path = (char *)malloc(directory_length + model_length + 1);
    if (*model_path != NULL)
    {
      /* Copy the directory path prefix. */
      memcpy(*model_path, manifest_path, directory_length);
      /* Append the model filename (including the null terminator). */
      memcpy(*model_path + directory_length, model, model_length + 1);
    }
  }

  if (*model_path == NULL)
  {
    SetError(error_message, error_message_size,
             "Memory allocation failed while resolving ONNX model path.");
    return false;
  }
  return true;
}

/**
 * @brief Parses input mapping objects into the owned manifest representation.
 * @param array Validated JSON array from the manifest root.
 * @param manifest Destination whose input array and strings become owned.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every input mapping satisfies the version 1 contract.
 *
 * On failure, the caller releases any partially populated entries through
 * OnnxAlquimiaFreeManifest.
 */
static bool ParseInputMappings(
    const cJSON *array,
    OnnxAlquimiaManifest *manifest,
    char *error_message,
    size_t error_message_size)
{
  static const char *const allowed[] = {
      "tensor", "tensor_element_index", "feature", "alquimia_state",
      "alquimia_state_index"};
  cJSON *item;
  size_t i = 0;
  /* In cJSON API, Array is key:[key : value,key : value,...]*/
  int count = cJSON_GetArraySize(array);

  /* Check if it's empty 
  ** Check if count * total input tensor exceeds the size_t
  */
  if (count < 0 || (size_t)count > SIZE_MAX / sizeof(*manifest->inputs))
  {
    SetError(error_message, error_message_size,
             "ONNX manifest input mapping array is too large.");
    return false;
  }
  manifest->num_inputs = (size_t)count;
  /* Check if there is key inside the inputs */
  if (count > 0)
  {
    manifest->inputs = (OnnxAlquimiaInputMappingSpec *)calloc(
        (size_t)count, sizeof(*manifest->inputs));
    if (manifest->inputs == NULL)
    {
      SetError(error_message, error_message_size,
               "Memory allocation failed for ONNX input mappings.");
      return false;
    }
  }
  /* Traverse all the elements in inputs */
  cJSON_ArrayForEach(item, array)
  {
    int tensor_element_index;
    if (!cJSON_IsObject(item) ||
        /* Check if there are duplicate keys */
        !ValidateProperties(item, allowed, 5, "input mapping",
                            error_message, error_message_size) ||
        /* Assign the value to the OnnxEngineStatus engine->manifest */
        /* Assign the value to the OnnxEngineStatus engine->manifest */
        !GetRequiredString(item, "tensor", "input mapping",
                           &manifest->inputs[i].tensor,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "tensor_element_index", "input mapping",
                            &tensor_element_index, error_message,
                            error_message_size) ||
        !GetRequiredString(item, "feature", "input mapping",
                           &manifest->inputs[i].feature,
                           error_message, error_message_size) ||
        !GetRequiredString(item, "alquimia_state", "input mapping",
                           &manifest->inputs[i].alquimia_state,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "alquimia_state_index", "input mapping",
                            &manifest->inputs[i].alquimia_state_index,
                            error_message, error_message_size))
    {
      if (!cJSON_IsObject(item))
      {
        SetError(error_message, error_message_size,
                 "Every ONNX input mapping must be an object.");
      }
      return false;
    }
    manifest->inputs[i].tensor_element_index =
        (size_t)tensor_element_index;
    ++i;
  }
  return true;
}

/**
 * @brief Parses output mapping objects into the owned manifest representation.
 * @param array Validated JSON array from the manifest root.
 * @param manifest Destination whose output array and strings become owned.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every output mapping satisfies the version 1 contract.
 *
 * Outputs omit feature names because conditions and problem metadata describe
 * model inputs only.
 */
static bool ParseOutputMappings(
    const cJSON *array,
    OnnxAlquimiaManifest *manifest,
    char *error_message,
    size_t error_message_size)
{
  static const char *const allowed[] = {
      "tensor", "tensor_element_index", "alquimia_state",
      "alquimia_state_index"};
  cJSON *item;
  size_t i = 0;
  int count = cJSON_GetArraySize(array);

  if (count < 0 || (size_t)count > SIZE_MAX / sizeof(*manifest->outputs))
  {
    SetError(error_message, error_message_size,
             "ONNX manifest output mapping array is too large.");
    return false;
  }
  manifest->num_outputs = (size_t)count;
  if (count > 0)
  {
    manifest->outputs = (OnnxAlquimiaOutputMappingSpec *)calloc(
        (size_t)count, sizeof(*manifest->outputs));
    if (manifest->outputs == NULL)
    {
      SetError(error_message, error_message_size,
               "Memory allocation failed for ONNX output mappings.");
      return false;
    }
  }

  cJSON_ArrayForEach(item, array)
  {
    int tensor_element_index;
    if (!cJSON_IsObject(item) ||
        !ValidateProperties(item, allowed, 4, "output mapping",
                            error_message, error_message_size) ||
        !GetRequiredString(item, "tensor", "output mapping",
                           &manifest->outputs[i].tensor,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "tensor_element_index", "output mapping",
                            &tensor_element_index, error_message,
                            error_message_size) ||
        !GetRequiredString(item, "alquimia_state", "output mapping",
                           &manifest->outputs[i].alquimia_state,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "alquimia_state_index", "output mapping",
                            &manifest->outputs[i].alquimia_state_index,
                            error_message, error_message_size))
    {
      if (!cJSON_IsObject(item))
      {
        SetError(error_message, error_message_size,
                 "Every ONNX output mapping must be an object.");
      }
      return false;
    }
    manifest->outputs[i].tensor_element_index =
        (size_t)tensor_element_index;
    ++i;
  }
  return true;
}

/**
 * @brief Reads a complete manifest file into a null-terminated buffer.
 * @param path Manifest filesystem path.
 * @param json_contents Receives newly allocated file json_contents.
 * @param length Receives the file length, excluding the terminator.
 * @param error_message Destination for read or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True on success; false after closing the file and freeing partial
 *         storage.
 *
 * This helper owns the FILE for its entire lifetime, keeping file cleanup out
 * of the parser's validation control flow.
 */
static bool ReadManifestjson_contents(
    const char *path,
    char **json_contents,
    size_t *length,
    char *error_message,
    size_t error_message_size)
{
  /* Opened file handle for the manifest JSON. */
  FILE *file;
  /* Number of bytes of the file */
  long file_size;
  size_t bytes_read;

  *json_contents = NULL;
  *length = 0;
  file = fopen(path, "rb");
  /* Open the file and determine its size by seeking, resetting the pointer afterward. */
  if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
      (file_size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0)
  {
    if (file != NULL)
    {
      fclose(file);
    }
    snprintf(error_message, error_message_size,
             "Unable to read ONNX manifest: %s", path);
    return false;
  }
  /* Check if the file size exceeds the size_t */
  if ((unsigned long)file_size > SIZE_MAX - 1)
  {
    fclose(file);
    SetError(error_message, error_message_size,
             "ONNX manifest file is too large.");
    return false;
  }

  *json_contents = (char *)malloc((size_t)file_size + 1);
  if (*json_contents == NULL)
  {
    fclose(file);
    SetError(error_message, error_message_size,
             "Memory allocation failed while reading ONNX manifest.");
    return false;
  }
  /* Number of bytes read from the JSON file. */
  /*  Returns the number of full items read. 
  ** This number may be less than count 
  ** if an error occurs or the end of the file is reached.
  */
  bytes_read = fread(*json_contents, 1, (size_t)file_size, file);
  fclose(file);
  /* Check if errors occur during reading JSON */
  if (bytes_read != (size_t)file_size)
  {
    free(*json_contents);
    *json_contents = NULL;
    SetError(error_message, error_message_size,
             "Failed to read the complete ONNX manifest.");
    return false;
  }

  (*json_contents)[bytes_read] = '\0';
  *length = bytes_read;
  return true;
}

/**
 * @brief Validates the root object and populates owned manifest storage.
 * @param path Manifest path used to resolve a relative model entry.
 * @param root Parsed cJSON root; ownership remains with the caller.
 * @param manifest Destination for copied paths and mapping specifications.
 * @param error_message Destination for schema or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when the root satisfies the complete version 1 contract.
 */
static bool PopulateManifest(
    const char *path,
    const cJSON *root,
    OnnxAlquimiaManifest *manifest,
    char *error_message,
    size_t error_message_size)
{
  /* The allowed keys in the first layer */
  static const char *const allowed[] = {
      "schema_version", "model", "inputs", "outputs"};
  const cJSON *schema_version;
  const cJSON *model;
  const cJSON *inputs;
  const cJSON *outputs;
  /* cJSON structure reference:
  ** - Object: { "key": value }
  ** - Array:  [ value, value ], for the value inside []: "key":value
  */
  if (!cJSON_IsObject(root))
  {
    SetError(error_message, error_message_size,
             "ONNX manifest root must be an object.");
    return false;
  }
  /* Check duplicate keys */
  if (!ValidateProperties(root, allowed, 4, "manifest root",
                          error_message, error_message_size))
  {
    return false;
  }

  schema_version = cJSON_GetObjectItemCaseSensitive(root, "schema_version");
  model = cJSON_GetObjectItemCaseSensitive(root, "model");
  inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
  outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
  /* In cJSON API, there is only double value */
  if (!cJSON_IsNumber(schema_version) || schema_version->valuedouble != 1.0)
  {
    SetError(error_message, error_message_size,
             "ONNX manifest schema_version must be integer 1.");
    return false;
  }
  if (!cJSON_IsString(model) || model->valuestring == NULL ||
      model->valuestring[0] == '\0')
  {
    SetError(error_message, error_message_size,
             "ONNX manifest model must be a nonempty string.");
    return false;
  }
  if (!cJSON_IsArray(inputs) || !cJSON_IsArray(outputs))
  {
    SetError(error_message, error_message_size,
             "ONNX manifest inputs and outputs must be arrays.");
    return false;
  }

  return ResolveModelPath(path, model->valuestring, &manifest->model_path,
                          error_message, error_message_size) &&
         ParseInputMappings(inputs, manifest, error_message,
                            error_message_size) &&
         ParseOutputMappings(outputs, manifest, error_message,
                             error_message_size);
}

/**
 * @brief Releases a complete or partially initialized manifest.
 * @param manifest Manifest whose owned paths, mappings, and strings are freed.
 *
 * The structure is zeroed after cleanup, making repeated cleanup and setup
 * failure paths safe.
 */
void OnnxAlquimiaFreeManifest(OnnxAlquimiaManifest *manifest)
{
  size_t i;
  if (manifest == NULL)
  {
    return;
  }
  free(manifest->model_path);
  for (i = 0; i < manifest->num_inputs; ++i)
  {
    free(manifest->inputs[i].tensor);
    free(manifest->inputs[i].feature);
    free(manifest->inputs[i].alquimia_state);
  }
  free(manifest->inputs);
  for (i = 0; i < manifest->num_outputs; ++i)
  {
    free(manifest->outputs[i].tensor);
    free(manifest->outputs[i].alquimia_state);
  }
  free(manifest->outputs);
  memset(manifest, 0, sizeof(*manifest));
}

/**
 * @brief Loads and strictly validates a version 1 ONNX sidecar manifest.
 * @param path JSON filesystem path.
 * @param manifest Receives owned model-path and mapping storage on success.
 * @param error_message Destination for the first read, parse, or schema error.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True on success; false after releasing all partial allocations.
 *
 * Parsing is length-aware, requires one complete JSON document, and rejects
 * unknown or duplicate properties. No cJSON-owned pointer escapes this call.
 */
bool OnnxAlquimiaLoadManifest(
    const char *path,
    OnnxAlquimiaManifest *manifest,
    char *error_message,
    size_t error_message_size)
{
  /* The onnx_alquimia_manifest checks the duplicate keys */
  /* onnx_alquimia_interface checks the duplicate values */
  char *json_contents = NULL;
  size_t bytes_read;
  const char *parse_end = NULL;
  cJSON *root = NULL;
  bool success;

  memset(manifest, 0, sizeof(*manifest));
  /* Check if the JSON path is valid (null/empty)*/
  if (path == NULL || path[0] == '\0')
  {
    /* Set up the error message for the AlquimiaEngineStatus: status */
    SetError(error_message, error_message_size,
             "ONNX manifest file path not provided.");
    return false;
  }
  if (!ReadManifestjson_contents(path, &json_contents, &bytes_read,
                            error_message, error_message_size))
  {
    return false;
  }

  /* Requiring the terminator and checking parse_end rejects otherwise-valid
  ** JSON followed by non-whitespace trailing content. */
  root = cJSON_ParseWithLengthOpts(
      json_contents, bytes_read + 1, &parse_end, true);
  if (root == NULL || parse_end == NULL ||
      (size_t)(parse_end - json_contents) != bytes_read)
  {
    SetError(error_message, error_message_size,
             "ONNX manifest is not valid strict JSON.");
    cJSON_Delete(root);
    free(json_contents);
    return false;
  }

  success = PopulateManifest(path, root, manifest,
                             error_message, error_message_size);
  cJSON_Delete(root);
  free(json_contents);
  if (!success)
  {
    OnnxAlquimiaFreeManifest(manifest);
  }
  return success;
}
