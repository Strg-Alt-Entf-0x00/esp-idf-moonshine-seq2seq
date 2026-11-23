set(COMPONENT_SRCS
    "src/moonshine_context.c"
    "src/moonshine_decoder.c"
    "src/moonshine_encoder.c"
    "src/moonshine_loader.c"
    "src/moonshine_ops.c"
    "src/moonshine_parallel.c"
    "src/loaders/moonshine_loader_core.c"
    "src/loaders/moonshine_loader_vfs.c"
    "src/moonshine_tokenizer.c"
)

# Conditionally add audio utils if enabled
if(CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS)
    list(APPEND COMPONENT_SRCS "src/moonshine_audio_utils.c")
    message(STATUS "Moonshine: Audio preprocessing utilities enabled")
endif()

set(COMPONENT_REQUIRES "")

# Add ESP-DSP only if audio utils are enabled
if(CONFIG_MOONSHINE_ENABLE_AUDIO_UTILS)
    list(APPEND COMPONENT_REQUIRES "esp-dsp")
endif()

idf_component_register(
    SRCS ${COMPONENT_SRCS}
    INCLUDE_DIRS
        "include"
    PRIV_INCLUDE_DIRS
        "src/internal"
    PRIV_REQUIRES
        esp_timer
        freertos
        heap
        vfs
    REQUIRES ${COMPONENT_REQUIRES}
        esp-idf-model-manager
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3)
