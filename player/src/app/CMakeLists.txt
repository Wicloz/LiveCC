add_executable(player
    main.cpp
    video_view.cpp
    audio_output.cpp
    ui.cpp
)

target_link_libraries(player PRIVATE ccmf_engine raylib player_warnings)
