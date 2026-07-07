add_executable(ccmf_player
    main.cpp
    video_view.cpp
    audio_output.cpp
    ui.cpp
)

target_link_libraries(ccmf_player PRIVATE ccmf_engine raylib ccmf_player_warnings)
