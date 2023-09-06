
# for arm_math.h
ASM_SOURCES = \
$(LIBDAISY_DIR)/Drivers/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.s

# for arm_math.h
C_SOURCES = $(wildcard ././libDaisy/Drivers/CMSIS/DSP/Source/*/*.c)

CFLAGS += -Wall  # Include all warnings
CFLAGS += -Os    #  Optimize for size

# for arm_math.h
C_INCLUDES += \
-I$(LIBDAISY_DIR)/Drivers/CMSIS/DSP/Include

# Project Name
TARGET = patch_cdcc

# Sources
CPP_SOURCES = patch_cdcc.cpp

# Library Locations
LIBDAISY_DIR = ./libDaisy/
DAISYSP_DIR = ./DaisySP/

# Core location, and generic Makefile.
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile
