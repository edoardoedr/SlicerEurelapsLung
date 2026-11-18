cmake_minimum_required(VERSION 3.28.0...3.28.0 FATAL_ERROR)

project(SlicerLungCollapse)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/edoardoedr/SlicerLungCollapse")
set(EXTENSION_CONTRIBUTORS "Edoardo De Rose (University of Calabria)")
set(EXTENSION_DESCRIPTION "Predicts the collapsed lung (pneumothorax) of a patient with a finite element model and shows the collapse as a sequence")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/edoardoedr/SlicerLungCollapse/main/SlicerLungCollapse.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(LungCollapse)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
