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

  cJSON_ArrayForEach(property, object)
  {
    if (property->string == NULL ||
        !IsAllowedProperty(property->string, allowed, num_allowed))
    {
      snprintf(error_message, error_message_size,
               "Unknown property '%s' in %s.",
               property->string == NULL ? "" : property->string, context);
      return false;
    }

    for (i = 0; i < num_allowed; ++i)
    {
      if (strcmp(property->string, allowed[i]) == 0)
      {
        const cJSON *other;
        int count = 0;
        cJSON_ArrayForEach(other, object)
        {
          if (other->string != NULL &&
              strcmp(other->string, allowed[i]) == 0)
          {
            ++count;
          }
        }
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
  const char *separator;
  size_t directory_length;
  size_t model_length = strlen(model);

  // Absolute path /usr/...
  if (model[0] == '/')
  {
    *model_path = CopyString(model);
  }
  // relative path models/.onnx
  else
  {
    separator = strrchr(manifest_path, '/');
    directory_length = separator == NULL ? 0 : (size_t)(separator - manifest_path) + 1;
    if (directory_length > SIZE_MAX - model_length - 1)
    {
      SetError(error_message, error_message_size,
               "Resolved ONNX model path is too long.");
      return false;
    }
    *model_path = (char *)malloc(directory_length + model_length + 1);
    if (*model_path != NULL)
    {
      memcpy(*model_path, manifest_path, directory_length);
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
      "tensor", "element", "feature", "field", "index"};
  cJSON *item;
  size_t i = 0;
  int count = cJSON_GetArraySize(array);

  if (count < 0 || (size_t)count > SIZE_MAX / sizeof(*manifest->inputs))
  {
    SetError(error_message, error_message_size,
             "ONNX manifest input mapping array is too large.");
    return false;
  }
  manifest->num_inputs = (size_t)count;
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

  cJSON_ArrayForEach(item, array)
  {
    int element;
    if (!cJSON_IsObject(item) ||
        !ValidateProperties(item, allowed, 5, "input mapping",
                            error_message, error_message_size) ||
        !GetRequiredString(item, "tensor", "input mapping",
                           &manifest->inputs[i].tensor,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "element", "input mapping", &element,
                            error_message, error_message_size) ||
        !GetRequiredString(item, "feature", "input mapping",
                           &manifest->inputs[i].feature,
                           error_message, error_message_size) ||
        !GetRequiredString(item, "field", "input mapping",
                           &manifest->inputs[i].field,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "index", "input mapping",
                            &manifest->inputs[i].index,
                            error_message, error_message_size))
    {
      if (!cJSON_IsObject(item))
      {
        SetError(error_message, error_message_size,
                 "Every ONNX input mapping must be an object.");
      }
      return false;
    }
    manifest->inputs[i].element = (size_t)element;
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
      "tensor", "element", "field", "index"};
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
    int element;
    if (!cJSON_IsObject(item) ||
        !ValidateProperties(item, allowed, 4, "output mapping",
                            error_message, error_message_size) ||
        !GetRequiredString(item, "tensor", "output mapping",
                           &manifest->outputs[i].tensor,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "element", "output mapping", &element,
                            error_message, error_message_size) ||
        !GetRequiredString(item, "field", "output mapping",
                           &manifest->outputs[i].field,
                           error_message, error_message_size) ||
        !GetRequiredInteger(item, "index", "output mapping",
                            &manifest->outputs[i].index,
                            error_message, error_message_size))
    {
      if (!cJSON_IsObject(item))
      {
        SetError(error_message, error_message_size,
                 "Every ONNX output mapping must be an object.");
      }
      return false;
    }
    manifest->outputs[i].element = (size_t)element;
    ++i;
  }
  return true;
}

/**
 * @brief Reads a complete manifest file into a null-terminated buffer.
 * @param path Manifest filesystem path.
 * @param contents Receives newly allocated file contents.
 * @param length Receives the file length, excluding the terminator.
 * @param error_message Destination for read or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True on success; false after closing the file and freeing partial
 *         storage.
 *
 * This helper owns the FILE for its entire lifetime, keeping file cleanup out
 * of the parser's validation control flow.
 */
static bool ReadManifestContents(
    const char *path,
    char **contents,
    size_t *length,
    char *error_message,
    size_t error_message_size)
{
  FILE *file;
  long file_size;
  size_t bytes_read;

  *contents = NULL;
  *length = 0;
  file = fopen(path, "rb");
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
  if ((unsigned long)file_size > SIZE_MAX - 1)
  {
    fclose(file);
    SetError(error_message, error_message_size,
             "ONNX manifest file is too large.");
    return false;
  }

  *contents = (char *)malloc((size_t)file_size + 1);
  if (*contents == NULL)
  {
    fclose(file);
    SetError(error_message, error_message_size,
             "Memory allocation failed while reading ONNX manifest.");
    return false;
  }
  bytes_read = fread(*contents, 1, (size_t)file_size, file);
  fclose(file);
  if (bytes_read != (size_t)file_size)
  {
    free(*contents);
    *contents = NULL;
    SetError(error_message, error_message_size,
             "Failed to read the complete ONNX manifest.");
    return false;
  }

  (*contents)[bytes_read] = '\0';
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
  static const char *const allowed[] = {
      "schema_version", "model", "inputs", "outputs"};
  const cJSON *schema_version;
  const cJSON *model;
  const cJSON *inputs;
  const cJSON *outputs;

  if (!cJSON_IsObject(root))
  {
    SetError(error_message, error_message_size,
             "ONNX manifest root must be an object.");
    return false;
  }
  if (!ValidateProperties(root, allowed, 4, "manifest root",
                          error_message, error_message_size))
  {
    return false;
  }

  schema_version = cJSON_GetObjectItemCaseSensitive(root, "schema_version");
  model = cJSON_GetObjectItemCaseSensitive(root, "model");
  inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
  outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
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
    free(manifest->inputs[i].field);
  }
  free(manifest->inputs);
  for (i = 0; i < manifest->num_outputs; ++i)
  {
    free(manifest->outputs[i].tensor);
    free(manifest->outputs[i].field);
  }
  free(manifest->outputs);
  memset(manifest, 0, sizeof(*manifest));
}

/**
 * @brief Loads and strictly validates a version 1 ONNX sidecar manifest.
 * @param path Manifest filesystem path.
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
  char *contents = NULL;
  size_t bytes_read;
  const char *parse_end = NULL;
  cJSON *root = NULL;
  bool success;

  memset(manifest, 0, sizeof(*manifest));
  if (path == NULL || path[0] == '\0')
  {
    SetError(error_message, error_message_size,
             "ONNX manifest file path not provided.");
    return false;
  }
  if (!ReadManifestContents(path, &contents, &bytes_read,
                            error_message, error_message_size))
  {
    return false;
  }

  /* Requiring the terminator and checking parse_end rejects otherwise-valid
  ** JSON followed by non-whitespace trailing content. */
  root = cJSON_ParseWithLengthOpts(
      contents, bytes_read + 1, &parse_end, true);
  if (root == NULL || parse_end == NULL ||
      (size_t)(parse_end - contents) != bytes_read)
  {
    SetError(error_message, error_message_size,
             "ONNX manifest is not valid strict JSON.");
    cJSON_Delete(root);
    free(contents);
    return false;
  }

  success = PopulateManifest(path, root, manifest,
                             error_message, error_message_size);
  cJSON_Delete(root);
  free(contents);
  if (!success)
  {
    OnnxAlquimiaFreeManifest(manifest);
  }
  return success;
}
