set(TARGET_NAME "TEST")

enable_testing()

find_package(GTest CONFIG REQUIRED)

file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE CPP_FILES "*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${H_FILES} ${CPP_FILES})

add_executable(${TARGET_NAME} ${H_FILES} ${CPP_FILES})

set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17 OUTPUT_NAME ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER Test)
set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "${DEBUG_ARGUES}")

target_link_libraries(${TARGET_NAME} 
      PRIVATE
            GTest::gtest GTest::gtest_main
)

target_include_directories(
  ${TARGET_NAME} 
  PUBLIC 
)

add_test(
    NAME ${TARGET_NAME}
    COMMAND ${TARGET_NAME}
)