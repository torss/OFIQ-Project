function(download_file url file)
    message(STATUS "Downloading '${file}'")
    file(DOWNLOAD ${url} ${file} 
        SHOW_PROGRESS
        STATUS download_status
        LOG download_log
    )
    
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        list(GET download_status 1 status_string)
        message(FATAL_ERROR "Error downloading ${file}: ${status_string}\n${download_log}")
    endif()

    message(STATUS "Extracting '${file}' to '${PROJECT_SOURCE_DIR}/extern'")
    file(ARCHIVE_EXTRACT 
        INPUT ${file}
        DESTINATION ${PROJECT_SOURCE_DIR}/extern
        VERBOSE
    )
    
    file(REMOVE ${file})
endfunction(download_file)