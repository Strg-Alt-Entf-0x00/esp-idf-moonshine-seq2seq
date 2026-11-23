idf_component_register(
    SRCS "src/audio_manager.cpp"
    INCLUDE_DIRS "include"
    REQUIRES
        esp_codec_dev
        driver
        esp_driver_i2c
        esp_driver_i2s
        esp_driver_gpio
        esp_timer
        esp_ringbuf
        nvs_flash
)