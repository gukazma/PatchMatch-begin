target_link_libraries(PatchMatch PUBLIC 
Eigen3::Eigen
opencv_ml opencv_dnn opencv_core opencv_flann opencv_highgui
glog::glog
colmap::colmap
)

set_target_properties(PatchMatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PatchMatch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(ENABLE_CUDA_DEBUG)
add_compile_options(PatchMatch
PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:-g -G>
)
endif()

target_link_libraries(colmap PUBLIC 
Eigen3::Eigen
opencv_ml opencv_dnn opencv_core opencv_flann opencv_highgui
glog::glog
colmap::colmap
)

set_target_properties(colmap PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(colmap PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(colmap PRIVATE __CUDACC__)
