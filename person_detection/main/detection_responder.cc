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

/*
 * SPDX-FileCopyrightText: 2019-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ostream>
#include <stdio.h>
#include <queue>
#include <string>
#include <algorithm>
#include <map>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/mcpwm.h"
#include "soc/mcpwm_periph.h"


// Configuraciones del SERVO
#define SERVO_PIN 12
#define BUTTON_PIN static_cast<gpio_num_t>(15)

int last_button_state = 0;
int locker_open = 0;

#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "driver/gpio.h"
#define BLINK_GPIO GPIO_NUM_4

#include "esp_main.h"
#if DISPLAY_SUPPORT
#include "image_provider.h"
#include "bsp/esp-bsp.h"

static uint8_t s_led_state = 0;

// Camera definition is always initialized to match the trained detection model: 96x96 pix
// That is too small for LCD displays, so we extrapolate the image to 192x192 pix
#define IMG_WD (96 * 2)
#define IMG_HT (96 * 2)

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *lata_indicator = NULL;
static lv_obj_t *label = NULL;

static void create_gui(void)
{
  bsp_display_start();
  bsp_display_backlight_on(); // Set display brightness to 100%
  bsp_display_lock(0);
  camera_canvas = lv_canvas_create(lv_scr_act());
  assert(camera_canvas);
  lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

  lata_indicator = lv_led_create(lv_scr_act());
  assert(lata_indicator);
  lv_obj_align(lata_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
  lv_led_set_color(lata_indicator, lv_palette_main(LV_PALETTE_GREEN));

  label = lv_label_create(lv_scr_act());
  assert(label);
  lv_label_set_text_static(label, "Lata detected");
  lv_obj_align_to(label, lata_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
  bsp_display_unlock();
}
#endif // DISPLAY_SUPPORT


// Function to initialize the MCPWM module for controlling the servo
void mcpwm_example_gpio_initialize(void) {
    printf("Initializing MCPWM servo control...\n");
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, SERVO_PIN);
}

// Function to calculate pulse width for a given angle
// static uint32_t servo_per_degree_init(uint32_t degree_of_rotation) {
//     uint32_t cal_pulsewidth = 0;
//     cal_pulsewidth = (SERVO_MIN_PULSEWIDTH + (((SERVO_MAX_PULSEWIDTH - SERVO_MIN_PULSEWIDTH) * (degree_of_rotation)) / (SERVO_MAX_DEGREE)));
//     return cal_pulsewidth;
// }



void servo_control(mcpwm_unit_t mcpwm_num, mcpwm_timer_t timer_num, float angle) {
    // Calculate pulse width (500us - 2500us) corresponding to the angle (-90° - 270°)
    uint32_t duty_us = (500 + ((angle + 90) / 360.0) * 2000);
    mcpwm_set_duty_in_us(mcpwm_num, timer_num, MCPWM_OPR_A, duty_us);
}

#include <queue>
#include <iostream>

std::queue<std::string> scores_queue;
std::string mostFrequentClasses[4];
static int index = 0;
int mostFrequentCount = 0;

std::string clave1_pos0 = "blank_score";
std::string clave1_pos1 = "blank_score";
std::string clave1_pos2 = "blank_score";
std::string clave1_pos3 = "blank_score";

std::string clave_locker1[4] = { clave1_pos0, clave1_pos1, clave1_pos2, clave1_pos3};


bool detectClasses() {
  mcpwm_example_gpio_initialize();
  // button_initialize();

  mcpwm_config_t pwm_config;
  pwm_config.frequency = 50;  // Frequency = 50Hz, i.e., 20ms period for servos
  pwm_config.cmpr_a = 0;      // Duty cycle of PWM0A = 0 (for servo 1)
  pwm_config.cmpr_b = 0;      // Duty cycle of PWM0B = 0 (for servo 2)
  pwm_config.counter_mode = MCPWM_UP_COUNTER;
  pwm_config.duty_mode = MCPWM_DUTY_MODE_0;
  mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &pwm_config); 


  printf("CLASES DETECTADAS: ");
  for (int i = 0; i < 4; ++i) {
    printf("%s ", mostFrequentClasses[i].c_str());
  }
  printf("\n");

  bool unlock1 = true;

  for (int i = 0; i<4; ++i) {
    printf("MOST FREQUENT CLASSES[i]: %s\n", mostFrequentClasses[i].c_str());
    printf("CLAVE LOCKER[i]: %s\n", clave_locker1[i].c_str());
    if (mostFrequentClasses[i] != clave_locker1[i]) {
      printf("CLAVE NO COINCIDE");
      unlock1 = false;
      break;
    }
  }

  if (unlock1 == true) {
    printf("ABRIR LOCKER\n");
    for(int angle=-90; angle<=270; angle++){
          servo_control(MCPWM_UNIT_0, MCPWM_TIMER_0, angle);
    }
    locker_open = 1;

    while(locker_open == 1) {
      int button_state = gpio_get_level(BUTTON_PIN);
      printf("BUTTON STATE: %d\n", button_state);

      if(button_state == 1 && last_button_state == 0){
        for(int angle=270; angle>=-90; angle--){
          servo_control(MCPWM_UNIT_0, MCPWM_TIMER_0, angle);
        }
        locker_open = 0;
      }
      last_button_state = button_state;
    }

    return true;
  } else {
    return false;
  }     
}

bool RespondToDetection(float first_score, float second_score, float third_score, float fourth_score, float fifth_score, float sixth_score, float blank_score) {
  int first_score_int = (first_score) * 100 + 0.5;
  int second_score_int = (second_score) * 100 + 0.5;
  int third_score_int = (third_score) * 100 + 0.5;
  int fourth_score_int = (fourth_score) * 100 + 0.5;
  int fifth_score_int = (fifth_score) * 100 + 0.5;
  int sixth_score_int = (sixth_score) * 100 + 0.5;
  int blank_score_int = (blank_score) * 100 + 0.5;

  int max_score_int = std::max({first_score_int, second_score_int, third_score_int, 
                                  fourth_score_int, fifth_score_int, sixth_score_int, blank_score_int});


  if (max_score_int == first_score_int) {
      scores_queue.push("first_score");
  } else if (max_score_int == second_score_int) {
      scores_queue.push("second_score");
  } else if (max_score_int == third_score_int) {
      scores_queue.push("third_score");
  } else if (max_score_int == fourth_score_int) {
      scores_queue.push("fourth_score");
  } else if (max_score_int == fifth_score_int) {
      scores_queue.push("fifth_score");
  } else if (max_score_int == sixth_score_int) {
      scores_queue.push("sixth_score");
  } else if (max_score_int == blank_score_int) {
      scores_queue.push("blank_score");
  }

  if (scores_queue.size() > 5) {
    scores_queue.pop();
  }

  if (scores_queue.size() == 5) {
    std::map<std::string, int> frequencyMap;

    std::queue<std::string> tempQueue = scores_queue;
    while(!tempQueue.empty()) {
      frequencyMap[tempQueue.front()]++;
      tempQueue.pop();
    }

    std::string mostFrequentClass;
    int maxCount = 0;
    for (const auto& entry : frequencyMap) {
      if (entry.second > maxCount) {
          maxCount = entry.second;
          mostFrequentClass = entry.first;
      }
    }

    mostFrequentClasses[index] = mostFrequentClass;
    index = (index + 1) % 4; // Avanzar al siguiente índice circular

    printf("CLASS DETECTED: %s\n", mostFrequentClass.c_str());

    if (index == 0) {
      // bool unlock_locker = detectClasses();
      detectClasses();
      std::fill(std::begin(mostFrequentClasses), std::end(mostFrequentClasses), "");
    }

    std::queue<std::string> emptyQueue;
    std::swap(scores_queue, emptyQueue);

    return true;

  }

#if DISPLAY_SUPPORT
    if (!camera_canvas) {
      create_gui();
    }

    uint16_t *buf = (uint16_t *) image_provider_get_display_buf();

    bsp_display_lock(0);
    lv_canvas_set_buffer(camera_canvas, buf, IMG_WD, IMG_HT, LV_IMG_CF_TRUE_COLOR);
    bsp_display_unlock();
#endif // DISPLAY_SUPPORT
  MicroPrintf("1 score: %d%%\n", first_score_int);
  MicroPrintf("2 score: %d%%\n", second_score_int);
  MicroPrintf("3 score: %d%%\n", third_score_int);
  MicroPrintf("4 score: %d%%\n", fourth_score_int);
  MicroPrintf("5 score: %d%%\n", fifth_score_int);
  MicroPrintf("6 score: %d%%\n", sixth_score_int);
  MicroPrintf("Blank score: %d%%\n", blank_score_int);

  return false;
}