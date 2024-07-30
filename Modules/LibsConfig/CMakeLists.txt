target_link_libraries(PatchMatch PUBLIC 
Eigen3::Eigen
opencv_ml opencv_dnn opencv_core opencv_flann opencv_highgui
freeimage::FreeImage freeimage::FreeImagePlus
rerun_sdk
glog::glog
colmap::colmap
)
