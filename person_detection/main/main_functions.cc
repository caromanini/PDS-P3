/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "driver/gpio.h"
#include "driver/mcpwm.h"
#include "esp_attr.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"

#include <stdio.h>
#include <string>

#define DETECTION_THRESHOLD 0.8
#define TIME_THRESHOLD_SEC 5
#define FRAME_RATE 10

#define BUTTON_PIN GPIO_NUM_15

static int last_button_state = 0;

int detection_count = 0;
int frame_count = 0;

// #include "tensorflow/lite/micro/kernels/esp_nn/conv_timer.h"
// #include "tensorflow/lite/micro/kernels/esp_nn/fully_connected_timer.h"
// #include "tensorflow/lite/micro/kernels/esp_nn/pooling_timer.h"
// #include "tensorflow/lite/micro/kernels/esp_nn/softmax_timer.h"
// #include "tensorflow/lite/micro/kernels/reshape.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 100 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
// constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
// constexpr int kTensorArenaSize = 165 * 1024 + scratchBufSize;
constexpr int kTensorArenaSize = 150 * 1024;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace


// Setup GPIO for buttons
static void button_initialize(void) {
    gpio_config_t io_conf;
    io_conf.intr_type = GPIO_INTR_DISABLE; // No interrupts
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE; // Enable pull-up for buttons
    io_conf.pin_bit_mask = (1ULL << BUTTON_PIN);
    gpio_config(&io_conf);
}

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  button_initialize();



#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() {
  while(1) {
    int button_state = gpio_get_level(BUTTON_PIN);

    if (button_state == 1 && last_button_state == 0) {
      // Get image from provider.
      if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
        MicroPrintf("Image capture failed.");
      }

      // Run the model on this input and make sure it succeeds.
      if (kTfLiteOk != interpreter->Invoke()) {
        MicroPrintf("Invoke failed.");
      }

      TfLiteTensor* output = interpreter->output(0);

      // Process the inference results.
      int8_t first_score = output->data.uint8[k1Index];
      int8_t second_score = output->data.uint8[k2Index];
      int8_t third_score = output->data.uint8[k3Index];
      int8_t fourth_score = output->data.uint8[k4Index];
      int8_t fifth_score = output->data.uint8[k5Index];
      int8_t sixth_score = output->data.uint8[k6Index];
      int8_t blank_score = output->data.uint8[kBlankIndex];

      float first_score_f =
          (first_score - output->params.zero_point) * output->params.scale;
      float second_score_f =
          (second_score - output->params.zero_point) * output->params.scale;
      float third_score_f =
          (third_score - output->params.zero_point) * output->params.scale;
      float fourth_score_f =
          (fourth_score - output->params.zero_point) * output->params.scale;
      float fifth_score_f =
          (fifth_score - output->params.zero_point) * output->params.scale;
      float sixth_score_f =
          (sixth_score - output->params.zero_point) * output->params.scale;
      float blank_score_f =
          (blank_score - output->params.zero_point) * output->params.scale;   
          
      // Respond to detection
      bool delay_code = RespondToDetection(first_score_f, second_score_f, third_score_f, fourth_score_f, fifth_score_f, sixth_score_f, blank_score_f);
      if (delay_code == true) {
        vTaskDelay(1);
      } else if (delay_code == false) {
        vTaskDelay(1);
      }  


    }
  }


  



}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

#include <queue>

// extern std::queue<float> scores_queue;
extern std::queue<std::string> scores_queue;

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;

    printf("%d, ", input->data.int8[i]);
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  float first_score = output->data.uint8[k1Index];
  float second_score = output->data.uint8[k2Index];
  float third_score = output->data.uint8[k3Index];
  float fourth_score = output->data.uint8[k4Index];
  float fifth_score = output->data.uint8[k5Index];
  float sixth_score = output->data.uint8[k6Index];
  float blank_score = output->data.uint8[kBlankIndex];

  float first_score_f =
      (first_score - output->params.zero_point) * output->params.scale;
  float second_score_f =
      (second_score - output->params.zero_point) * output->params.scale;
  float third_score_f =
      (third_score - output->params.zero_point) * output->params.scale;
  float fourth_score_f =
      (fourth_score - output->params.zero_point) * output->params.scale;
  float fifth_score_f =
      (fifth_score - output->params.zero_point) * output->params.scale;
  float sixth_score_f =
      (sixth_score - output->params.zero_point) * output->params.scale;
  float blank_score_f =
      (blank_score - output->params.zero_point) * output->params.scale;    

  RespondToDetection(first_score_f, second_score_f, third_score_f, fourth_score_f, fifth_score_f, sixth_score_f, blank_score_f);

  frame_count += 1;
}