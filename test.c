/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32f1xx_hal.h"
#include "stm32f1xx_hal_tim.h"
#include "stm32f1xx_hal_gpio.h"
#include "main.h"
#include <stdio.h>
#include <stdint.h>
#include "stm32f1xx.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
// 自定义类型定义
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
/* USER CODE END Includes */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// 舵机参数（50Hz PWM，20ms周期）
#define SERVO_MIN_CCR    5     // 0°对应CCR值（0.5ms高电平）
#define SERVO_MAX_CCR    25    // 180°对应CCR值（2.5ms高电平）
#define SERVO_MID_CCR    15    // 90°对应CCR值（1.5ms高电平）

// 推进电机参数（1kHz PWM，1ms周期）
#define MOTOR_MAX_CCR    999   // 100%占空比对应CCR值
#define MOTOR_MIN_CCR    0     // 0%占空比对应CCR值
/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
u16 Servo_Current_CCR = SERVO_MID_CCR; // 当前舵机CCR值（默认90°）
u16 Motor_Current_CCR = MOTOR_MIN_CCR; // 当前电机CCR值（默认停转）
TIM_HandleTypeDef htim2;  // TIM2句柄（仅在main.c中定义，无重复）
TIM_HandleTypeDef htim3;  // TIM3句柄（仅在main.c中定义，无重复）
/* USER CODE END PV */

/* Private function prototypes -------