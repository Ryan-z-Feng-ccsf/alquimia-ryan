/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

#include "alquimia/onnx_alquimia_config.h"

#include <limits.h>
#include <math.h>
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
 * @param error_message Destination buffer, which may be NULL.
 * @param error_message_size Size of @p message in bytes.
 * @param detail Null-terminated error detail.
 */
static void SetError(char *error_message, size_t error_message_size, const char *error_detail)
{
  if (error_message != NULL && error_message_size > 0)
  {
    snprintf(error_message, error_message_size, "%s", error_detail);
  }
}

/**
 * @brief Creates a C-allocator-owned copy of a parsed JSON string.
 * @param str_value Null-terminated source string owned by cJSON.
 * @return Newly allocated storage, or NULL on allocation failure.
 */
static char *CopyString(const char *str_value)
{
  size_t length = strlen(str_value);
  char *copy = (char *)malloc(length + 1);
  if (copy != NULL)
  {
    memcpy(copy, str_value, length + 1);
  }
  return copy;
}

/**
 * @brief Checks a property name against an exact, case-sensitive allowlist.
 * @param name Property name to check.
 * @param allowed_schema_fields Accepted property names.
 * @param num_allowed_schema_fields Number of entries in @p allowed.
 * @return True only for an exact allowlist match.
 */
static bool IsAllowedProperty(
    const char *name,
    const char *const *allowed_schema_fields,
    size_t num_allowed_schema_fields)
{
  size_t i;
  for (i = 0; i < num_allowed_schema_fields; ++i)
  {
    if (strcmp(name, allowed_schema_fields[i]) == 0)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief Rejects unknown and duplicate members in one config object.
 * @param cjson_object cJSON object whose direct children are inspected.
 * @param allowed_schema_fields Exact property allowlist for this object type.
 * @param num_allowed_schema_fields Number of entries in @p allowed_schema_fields.
 * @param context Human-readable object name used in errors.
 * @param error_message Destination for the first validation error.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every member is allowed_schema_fields and appears at most once.
 *
 * cJSON permits duplicate object keys and lookup returns only one occurrence,
 * so strict validation must traverse the complete child list explicitly.
 */
static bool ValidateProperties(
    const cJSON *cjson_object,
    const char *const *allowed_schema_fields,
    size_t num_allowed_schema_fields,
    const char *context,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_property;
  size_t i;

  /* Triple loop to check duplicate keys */
  /* Trade-off between space and time performance */
  /* Right here, pick the space */
  /* Uses an O(N^2 * M) triple loop to detect duplicate keys without allocating 
  ** extra memory, prioritizing space efficiency over time. */
  cJSON_ArrayForEach(cjson_property, cjson_object)
  {
    /* Check if the key is empty */
    /* Check if the key is valid */
    if (cjson_property->string == NULL ||
        !IsAllowedProperty(cjson_property->string, allowed_schema_fields, num_allowed_schema_fields))
    {
      snprintf(error_message, error_message_size,
               "Unknown property '%s' in %s.",
               cjson_property->string == NULL ? "" : cjson_property->string, context);
      return false;
    }
    /* Second layer of the loop, traverse the allowed_schema_fields keys */
    for (i = 0; i < num_allowed_schema_fields; ++i)
    {
      /* If the key was found */
      if (strcmp(cjson_property->string, allowed_schema_fields[i]) == 0)
      {
        /* A new cJSON used to find the duplicate key */
        const cJSON *cjson_other;
        /* Record the number of each key */
        int count = 0;
        /* Traverse the key from the start to record the number of each key */
        cJSON_ArrayForEach(cjson_other, cjson_object)
        {
          if (cjson_other->string != NULL &&
              strcmp(cjson_other->string, allowed_schema_fields[i]) == 0)
          {
            ++count;
          }
        }
        /* If the count > 1, duplicate key */
        if (count > 1)
        {
          snprintf(error_message, error_message_size,
                   "Duplicate property '%s' in %s.", allowed_schema_fields[i], context);
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
 * @param cjson_object Object containing the property.
 * @param name Exact property name.
 * @param context Human-readable object name used in errors.
 * @param str_value Receives C-allocator-owned storage on success.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when the property is a nonempty string and copying succeeds.
 *
 * Copying detaches the config representation from the cJSON tree lifetime.
 */
static bool GetRequiredString(
    const cJSON *cjson_object,
    const char *name,
    const char *context,
    char **str_value,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_item = cJSON_GetObjectItemCaseSensitive(cjson_object, name);
  if (!cJSON_IsString(cjson_item) || cjson_item->valuestring == NULL ||
      cjson_item->valuestring[0] == '\0')
  {
    snprintf(error_message, error_message_size,
             "Required property '%s' in %s must be a nonempty string.",
             name, context);
    return false;
  }
  *str_value = CopyString(cjson_item->valuestring);
  if (*str_value == NULL)
  {
    SetError(error_message, error_message_size,
             "Memory allocation failed while loading ONNX config.");
    return false;
  }
  return true;
}

/**
 * @brief Reads a required integer representable as a nonnegative C int.
 * @param cjson_object Object containing the property.
 * @param name Exact property name.
 * @param context Human-readable object name used in errors.
 * @param int_value Receives the parsed integer on success.
 * @param error_message Destination for validation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when the JSON number is integral and lies in [0, INT_MAX].
 *
 * cJSON stores numbers as double, so the round-trip comparison rejects
 * fractional values before conversion to the config's integer fields.
 */
static bool GetRequiredInteger(
    const cJSON *cjson_object,
    const char *name,
    const char *context,
    int *int_value,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_item = cJSON_GetObjectItemCaseSensitive(cjson_object, name);
  if (!cJSON_IsNumber(cjson_item) || cjson_item->valuedouble < 0.0 ||
      cjson_item->valuedouble > INT_MAX ||
      (double)(int)cjson_item->valuedouble != cjson_item->valuedouble)
  {
    snprintf(error_message, error_message_size,
             "Required property '%s' in %s must be a nonnegative integer.",
             name, context);
    return false;
  }
  *int_value = (int)cjson_item->valuedouble;
  return true;
}

/**
 * @brief Resolves the config model entry to an owned filesystem path.
 * @param config_path Path used to open the config.
 * @param model Nonempty model entry from the config.
 * @param model_path Receives newly allocated path storage.
 * @param error_message Destination for overflow or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when path construction succeeds.
 *
 * Absolute paths are preserved. Relative paths use the config's directory,
 * not the process working directory; existence is checked later by setup.
 */
static bool ResolveModelPath(
    const char *config_path,
    const char *model,
    char **model_path,
    char *error_message,
    size_t error_message_size)
{
  /* A pointer points to the last '/' */
  const char *last_separator;
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
    last_separator = strrchr(config_path, '/');
    /* Get the length of the directory inclding the last '/' */
    directory_length = last_separator == NULL ? 0 : (size_t)(last_separator - config_path) + 1;
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
      memcpy(*model_path, config_path, directory_length);
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
 * @brief Parses input mapping objects into the owned config representation.
 * @param cjson_array Validated JSON array from the config root.
 * @param onnx_config Destination whose input array and strings become owned.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every input mapping satisfies the version 1 contract.
 *
 * On failure, the caller releases any partially populated entries through
 * OnnxAlquimiaFreeConfig.
 */
static bool ParseInputMappings(
    const cJSON *cjson_array,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  static const char *const allowed_schema_fields[] = {
      "tensor", "tensor_element_index", "feature", "alquimia_state",
      "alquimia_state_index"};
  cJSON *cjson_item;
  size_t i = 0;
  /* In cJSON API, Array is key:[key : value,key : value,...]*/
  int count = cJSON_GetArraySize(cjson_array);

  /* Check if it's empty 
  ** Check if count * total input tensor exceeds the size_t
  */
  if (count < 0 || (size_t)count > SIZE_MAX / sizeof(*onnx_config->inputs))
  {
    SetError(error_message, error_message_size,
             "ONNX config input mapping array is too large.");
    return false;
  }
  onnx_config->num_inputs = (size_t)count;
  /* Check if there is key inside the inputs */
  if (count > 0)
  {
    onnx_config->inputs = (OnnxAlquimiaInputMapping *)calloc(
        (size_t)count, sizeof(*onnx_config->inputs));
    if (onnx_config->inputs == NULL)
    {
      SetError(error_message, error_message_size,
               "Memory allocation failed for ONNX input mappings.");
      return false;
    }
  }
  /* Traverse all the elements in inputs */
  cJSON_ArrayForEach(cjson_item, cjson_array)
  {
    int tensor_element_index;
    if (!cJSON_IsObject(cjson_item) ||
        /* Check if there are duplicate keys */
        !ValidateProperties(cjson_item, allowed_schema_fields, 5, "input mapping",
                            error_message, error_message_size) ||
        /* Assign the value to the OnnxEngineStatus engine->config->inputMapping */
        /* Assign the value to the OnnxEngineStatus engine->config->inputMapping */
        !GetRequiredString(cjson_item, "tensor", "input mapping",
                           &onnx_config->inputs[i].tensor,
                           error_message, error_message_size) ||
        !GetRequiredInteger(cjson_item, "tensor_element_index", "input mapping",
                            &tensor_element_index, error_message,
                            error_message_size) ||
        !GetRequiredString(cjson_item, "feature", "input mapping",
                           &onnx_config->inputs[i].feature,
                           error_message, error_message_size) ||
        !GetRequiredString(cjson_item, "alquimia_state", "input mapping",
                           &onnx_config->inputs[i].alquimia_state,
                           error_message, error_message_size) ||
        !GetRequiredInteger(cjson_item, "alquimia_state_index", "input mapping",
                            &onnx_config->inputs[i].alquimia_state_index,
                            error_message, error_message_size))
    {
      if (!cJSON_IsObject(cjson_item))
      {
        SetError(error_message, error_message_size,
                 "Every ONNX input mapping must be an object.");
      }
      return false;
    }
    onnx_config->inputs[i].tensor_element_index =
        (size_t)tensor_element_index;
    ++i;
  }
  return true;
}

/**
 * @brief Rejects empty or duplicate arbitrary member names in an object.
 * @param cjson_object Object whose direct members are checked.
 * @param context Human-readable object name used in errors.
 * @param error_message Destination for the first validation error.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every direct member has a unique, nonempty name.
 *
 * Condition and feature names are user-defined, so they cannot use the fixed
 * schema allowlist enforced by ValidateProperties.
 */
static bool ValidateUniqueMember(
    const cJSON *cjson_object,
    const char *context,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_property;

  // Traverse the cjson_object(loop, O(N^2))
  cJSON_ArrayForEach(cjson_property, cjson_object)
  {
    const cJSON *cjson_other;
    if (cjson_property->string == NULL || cjson_property->string[0] == '\0')
    {
      snprintf(error_message, error_message_size,
               "Every name in %s must be nonempty.", context);
      return false;
    }
    // Find the duplicate element
    // Similar with the Linked List
    for (cjson_other = cjson_property->next;
         cjson_other != NULL;
         cjson_other = cjson_other->next)
    {
      if (cjson_other->string != NULL &&
          strcmp(cjson_property->string, cjson_other->string) == 0)
      {
        snprintf(error_message, error_message_size,
                 "Duplicate name '%s' in %s.",
                 cjson_property->string, context);
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Verifies that every condition supplies every mapped input feature.
 * @param onnx_config Config containing parsed inputs and conditions.
 * @param error_message Destination for the first validation error.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when all mapped features exist in each condition.
 *
 * Conditions may contain additional features so a shared condition can serve
 * models with different input subsets.
 */
static bool ValidateConditionFeature(
    const OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  size_t condition_index;
  size_t input_index;

  for (condition_index = 0;
       condition_index < onnx_config->num_conditions;
       ++condition_index)
  {
    const OnnxAlquimiaCondition *condition =
        &onnx_config->conditions[condition_index];
    for (input_index = 0; input_index < onnx_config->num_inputs; ++input_index)
    {
      const char *required_feature = onnx_config->inputs[input_index].feature;
      size_t item_index;
      bool found = false;
      for (item_index = 0; item_index < condition->num_items; ++item_index)
      {
        if (strcmp(condition->items[item_index].feature,
                   required_feature) == 0)
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        snprintf(error_message, error_message_size,
                 "Condition '%s' is missing input feature '%s'.",
                 condition->name, required_feature);
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Parses optional named JSON initial conditions.
 * @param cjson_conditions Conditions object from the config root.
 * @param onnx_config Destination whose condition storage becomes owned.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when all names and numeric feature values are valid.
 */
static bool ParseConditions(
    const cJSON *cjson_conditions,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_condition;
  size_t condition_index = 0;
  // Record the numbers of conditions
  int condition_count;

  // Driver could provide the condition
  if (cjson_conditions == NULL)
  {
    return true;
  }
  
  // "conditions" is an object
  /* "conditions": {
    "initial": {
      "Mineral_source": 7,
      "uranium_total": -6.677780705266080,
      "Site_Density": -4.54327863489071,
      "U_species1": -25.419000000000000,
      "U_species8": -22.514000000000000,
      "U_species14": -37.488999999999997,
      "U_species20": -20.510000000000002
    }
  } */
  if (!cJSON_IsObject(cjson_conditions))
  {
    SetError(error_message, error_message_size,
             "ONNX config conditions must be an object.");
    return false;
  }
  // Validate the name such as "initial"
  if (!ValidateUniqueMember(cjson_conditions, "conditions",
                            error_message, error_message_size))
  {
    return false;
  } 

  // Can be used for object/array
  condition_count = cJSON_GetArraySize(cjson_conditions);

  // Check the edge
  if (condition_count < 0 ||
    // Check if total conditions exceed the boundary
      (size_t)condition_count >
          SIZE_MAX / sizeof(*onnx_config->conditions))
  {
    SetError(error_message, error_message_size,
             "ONNX config conditions object is too large.");
    return false;
  }
  onnx_config->num_conditions = (size_t)condition_count;
  if (condition_count > 0)
  {
    onnx_config->conditions = (OnnxAlquimiaCondition *)calloc(
        (size_t)condition_count, sizeof(*onnx_config->conditions));
    if (onnx_config->conditions == NULL)
    {
      SetError(error_message, error_message_size,
               "Memory allocation failed for ONNX conditions.");
      return false;
    }
  }

  // Traverse the members of conditions
  // "initial" or other names...
  cJSON_ArrayForEach(cjson_condition, cjson_conditions)
  {
    OnnxAlquimiaCondition *condition =
        &onnx_config->conditions[condition_index];
    const cJSON *cjson_value;
    size_t item_index = 0;
    int item_count;

    condition->name = CopyString(cjson_condition->string);
    if (condition->name == NULL)
    {
      SetError(error_message, error_message_size,
               "Memory allocation failed for an ONNX condition name.");
      return false;
    }

    // The member of "conditions" should be an object
    if (!cJSON_IsObject(cjson_condition))
    {
      snprintf(error_message, error_message_size,
               "Condition '%s' must be an object.", condition->name);
      return false;
    }

    // Validate the elements like "Mineral_source"
    if (!ValidateUniqueMember(cjson_condition, "condition feature values",
                              error_message, error_message_size))
    {
      return false;
    }

    // Count the number of items in the "initial"
    item_count = cJSON_GetArraySize(cjson_condition);
    if (item_count < 0 ||
        // Check if the condition features exceed the boundary
        (size_t)item_count > SIZE_MAX / sizeof(*condition->items))
    {
      snprintf(error_message, error_message_size,
               "Condition '%s' has too many feature values.",
               condition->name);
      return false;
    }
    condition->num_items = (size_t)item_count;
    if (item_count > 0)
    {
      condition->items = (OnnxAlquimiaConditionItem *)calloc(
          (size_t)item_count, sizeof(*condition->items));
      if (condition->items == NULL)
      {
        SetError(error_message, error_message_size,
                 "Memory allocation failed for ONNX condition values.");
        return false;
      }
    }
    // Traverse the members of the condition
    cJSON_ArrayForEach(cjson_value, cjson_condition)
    {
      if (!cJSON_IsNumber(cjson_value) ||
          !isfinite(cjson_value->valuedouble))
      {
        snprintf(error_message, error_message_size,
                 "Feature '%s' in condition '%s' must be a finite number.",
                 cjson_value->string, condition->name);
        return false;
      }
      condition->items[item_index].feature =
          CopyString(cjson_value->string);
      if (condition->items[item_index].feature == NULL)
      {
        SetError(error_message, error_message_size,
                 "Memory allocation failed for an ONNX condition feature.");
        return false;
      }
      condition->items[item_index].value = cjson_value->valuedouble;
      ++item_index;
    }
    ++condition_index;
  }

  // Validate the features in the JSON 
  return ValidateConditionFeature(
      onnx_config, error_message, error_message_size);
}

/**
 * @brief Parses output mapping objects into the owned config representation.
 * @param cjson_array Validated JSON array from the config root.
 * @param onnx_config Destination whose output array and strings become owned.
 * @param error_message Destination for validation or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when every output mapping satisfies the version 1 contract.
 *
 * Outputs omit feature names because conditions and problem metadata describe
 * model inputs only.
 */
static bool ParseOutputMappings(
    const cJSON *cjson_array,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  static const char *const allowed_schema_fields[] = {
      "tensor", "tensor_element_index", "alquimia_state",
      "alquimia_state_index"};
  cJSON *cjson_item;
  size_t i = 0;
  int count = cJSON_GetArraySize(cjson_array);

  if (count < 0 || (size_t)count > SIZE_MAX / sizeof(*onnx_config->outputs))
  {
    SetError(error_message, error_message_size,
             "ONNX config output mapping array is too large.");
    return false;
  }
  onnx_config->num_outputs = (size_t)count;
  if (count > 0)
  {
    onnx_config->outputs = (OnnxAlquimiaOutputMapping *)calloc(
        (size_t)count, sizeof(*onnx_config->outputs));
    if (onnx_config->outputs == NULL)
    {
      SetError(error_message, error_message_size,
               "Memory allocation failed for ONNX output mappings.");
      return false;
    }
  }

  cJSON_ArrayForEach(cjson_item, cjson_array)
  {
    int tensor_element_index;
    if (!cJSON_IsObject(cjson_item) ||
        !ValidateProperties(cjson_item, allowed_schema_fields, 4, "output mapping",
                            error_message, error_message_size) ||
        !GetRequiredString(cjson_item, "tensor", "output mapping",
                           &onnx_config->outputs[i].tensor,
                           error_message, error_message_size) ||
        !GetRequiredInteger(cjson_item, "tensor_element_index", "output mapping",
                            &tensor_element_index, error_message,
                            error_message_size) ||
        !GetRequiredString(cjson_item, "alquimia_state", "output mapping",
                           &onnx_config->outputs[i].alquimia_state,
                           error_message, error_message_size) ||
        !GetRequiredInteger(cjson_item, "alquimia_state_index", "output mapping",
                            &onnx_config->outputs[i].alquimia_state_index,
                            error_message, error_message_size))
    {
      if (!cJSON_IsObject(cjson_item))
      {
        SetError(error_message, error_message_size,
                 "Every ONNX output mapping must be an object.");
      }
      return false;
    }
    onnx_config->outputs[i].tensor_element_index =
        (size_t)tensor_element_index;
    ++i;
  }
  return true;
}

/**
 * @brief Reads a complete config file into a null-terminated buffer.
 * @param config_path config filesystem path.
 * @param json_contents Receives newly allocated file json_contents.
 * @param file_length Receives the file length, excluding the terminator.
 * @param error_message Destination for read or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True on success; false after closing the file and freeing partial
 *         storage.
 *
 * This helper owns the FILE for its entire lifetime, keeping file cleanup out
 * of the parser's validation control flow.
 */
static bool ReadConfigjson_contents(
    const char *config_path,
    char **json_contents,
    size_t *file_length,
    char *error_message,
    size_t error_message_size)
{
  /* Opened file handle for the config JSON. */
  FILE *file;
  /* Number of bytes of the file */
  long file_size;
  size_t bytes_read;

  *json_contents = NULL;
  *file_length = 0;
  file = fopen(config_path, "rb");
  /* Open the file and determine its size by seeking, resetting the pointer afterward. */
  if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
      (file_size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0)
  {
    if (file != NULL)
    {
      fclose(file);
    }
    snprintf(error_message, error_message_size,
             "Unable to read ONNX config: %s", config_path);
    return false;
  }
  /* Check if the file size exceeds the size_t */
  if ((unsigned long)file_size > SIZE_MAX - 1)
  {
    fclose(file);
    SetError(error_message, error_message_size,
             "ONNX config file is too large.");
    return false;
  }

  *json_contents = (char *)malloc((size_t)file_size + 1);
  if (*json_contents == NULL)
  {
    fclose(file);
    SetError(error_message, error_message_size,
             "Memory allocation failed while reading ONNX config.");
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
             "Failed to read the complete ONNX config.");
    return false;
  }

  (*json_contents)[bytes_read] = '\0';
  *file_length = bytes_read;
  return true;
}

/**
 * @brief Validates the root object and populates owned config storage.
 * @param config_path config path used to resolve a relative model entry.
 * @param cjson_root Parsed cJSON root; ownership remains with the caller.
 * @param onnx_config Destination for copied paths and mapping specifications.
 * @param error_message Destination for schema or allocation errors.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True when the root satisfies the complete version 1 contract.
 */
static bool PopulateConfig(
    const char *config_path,
    const cJSON *cjson_root,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  /* The allowed keys in the first layer */
  static const char *const allowed_schema_fields[] = {
      "schema_version", "model", "conditions", "inputs", "outputs"};
  const cJSON *cjson_schema_version;
  const cJSON *cjson_model;
  const cJSON *cjson_conditions;
  const cJSON *cjson_inputs;
  const cJSON *cjson_outputs;
  /* cJSON structure reference:
  ** - Object: { "key": value }
  ** - Array:  [ value, value ], for the value inside []: "key":value
  */
  if (!cJSON_IsObject(cjson_root))
  {
    SetError(error_message, error_message_size,
             "ONNX config root must be an object.");
    return false;
  }
  /* Check duplicate keys */
  if (!ValidateProperties(cjson_root, allowed_schema_fields, 5, "config root",
                          error_message, error_message_size))
  {
    return false;
  }

  cjson_schema_version = cJSON_GetObjectItemCaseSensitive(cjson_root, "schema_version");
  cjson_model = cJSON_GetObjectItemCaseSensitive(cjson_root, "model");
  cjson_conditions = cJSON_GetObjectItemCaseSensitive(cjson_root, "conditions");
  cjson_inputs = cJSON_GetObjectItemCaseSensitive(cjson_root, "inputs");
  cjson_outputs = cJSON_GetObjectItemCaseSensitive(cjson_root, "outputs");
  /* In cJSON API, there is only double value */
  if (!cJSON_IsNumber(cjson_schema_version) || cjson_schema_version->valuedouble != 1.0)
  {
    SetError(error_message, error_message_size,
             "ONNX config schema_version must be integer 1.");
    return false;
  }
  if (!cJSON_IsString(cjson_model) || cjson_model->valuestring == NULL ||
      cjson_model->valuestring[0] == '\0')
  {
    SetError(error_message, error_message_size,
             "ONNX config model must be a nonempty string.");
    return false;
  }
  if (!cJSON_IsArray(cjson_inputs) || !cJSON_IsArray(cjson_outputs))
  {
    SetError(error_message, error_message_size,
             "ONNX config inputs and outputs must be arrays.");
    return false;
  }
  /* Set values for the OnnxEngineStatus engine->config */
  /* engine->config->model_path, engine
  ** engine->config->inputMapping
  ** engine->config->outpuMapping
  ** The input mapping is almost the same as the output mapping
  ** But the output mapping doesn't have the output feature
  */
  return ResolveModelPath(config_path, cjson_model->valuestring, &onnx_config->model_path,
                          error_message, error_message_size) &&
         ParseInputMappings(cjson_inputs, onnx_config, error_message,
                            error_message_size) &&
         ParseOutputMappings(cjson_outputs, onnx_config, error_message,
                             error_message_size) &&
         ParseConditions(cjson_conditions, onnx_config, error_message,
                         error_message_size);
}

/**
 * @brief Releases a complete or partially initialized config.
 * @param onnx_config config whose owned paths, mappings, and strings are freed.
 *
 * The structure is zeroed after cleanup, making repeated cleanup and setup
 * failure paths safe.
 */
void OnnxAlquimiaFreeConfig(OnnxAlquimiaConfig *onnx_config)
{
  size_t i;
  if (onnx_config == NULL)
  {
    return;
  }
  free(onnx_config->model_path);
  for (i = 0; i < onnx_config->num_conditions; ++i)
  {
    size_t j;
    free(onnx_config->conditions[i].name);
    for (j = 0; j < onnx_config->conditions[i].num_items; ++j)
    {
      free(onnx_config->conditions[i].items[j].feature);
    }
    free(onnx_config->conditions[i].items);
  }
  free(onnx_config->conditions);
  for (i = 0; i < onnx_config->num_inputs; ++i)
  {
    free(onnx_config->inputs[i].tensor);
    free(onnx_config->inputs[i].feature);
    free(onnx_config->inputs[i].alquimia_state);
  }
  free(onnx_config->inputs);
  for (i = 0; i < onnx_config->num_outputs; ++i)
  {
    free(onnx_config->outputs[i].tensor);
    free(onnx_config->outputs[i].alquimia_state);
  }
  free(onnx_config->outputs);
  memset(onnx_config, 0, sizeof(*onnx_config));
}

/**
 * @brief Loads and strictly validates a versioned ONNX sidecar config.
 * @param config_path JSON filesystem path.
 * @param onnx_config Receives owned model-path and mapping storage on success.
 * @param error_message Destination for the first read, parse, or schema error.
 * @param error_message_size Size of @p error_message in bytes.
 * @return True on success; false after releasing all partial allocations.
 *
 * Parsing is length-aware, requires one complete JSON document, and rejects
 * unknown or duplicate properties. No cJSON-owned pointer escapes this call.
 */
bool OnnxAlquimiaLoadConfig(
    const char *config_path,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  /* The onnx_alquimia_config checks the duplicate keys */
  /* onnx_alquimia_interface checks the duplicate values */
  char *json_contents = NULL;
  size_t bytes_read;
  const char *parse_end = NULL;
  cJSON *cjson_root = NULL;
  bool success;

  memset(onnx_config, 0, sizeof(*onnx_config));
  /* Check if the JSON path is valid (null/empty)*/
  if (config_path == NULL || config_path[0] == '\0')
  {
    /* Set up the error message for the AlquimiaEngineStatus: status */
    SetError(error_message, error_message_size,
             "ONNX config file path not provided.");
    return false;
  }
  if (!ReadConfigjson_contents(config_path, &json_contents, &bytes_read,
                            error_message, error_message_size))
  {
    return false;
  }

  /* Requiring the terminator and checking parse_end rejects otherwise-valid
  ** JSON followed by non-whitespace trailing content. */
  cjson_root = cJSON_ParseWithLengthOpts(
      json_contents, bytes_read + 1, &parse_end, true);
  if (cjson_root == NULL || parse_end == NULL ||
      (size_t)(parse_end - json_contents) != bytes_read)
  {
    SetError(error_message, error_message_size,
             "ONNX config is not valid strict JSON.");
    cJSON_Delete(cjson_root);
    free(json_contents);
    return false;
  }

  success = PopulateConfig(config_path, cjson_root, onnx_config,
                             error_message, error_message_size);
  cJSON_Delete(cjson_root);
  free(json_contents);
  if (!success)
  {
    OnnxAlquimiaFreeConfig(onnx_config);
  }
  return success;
}
