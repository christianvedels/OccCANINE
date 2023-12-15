# Setup_environment
# Created:  2023-12-15
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Sets up the correct python environment

# Load reticulate into current R session
library(reticulate)
library(here)

# Create environment if it does not exist
conda_envs = conda_list()
if(!any("r-HISCO" %in% conda_envs$name)){
  conda_create("r-HISCO")
}

# Use environment
use_condaenv("r-HISCO")

# For initialization
reticulate::py_config()
# Check if python is available
reticulate::py_available()

# Get list of installed packages
package = py_list_packages()$package

# List required packages
required_packages = c(
  "transformers",
  "numpy"
)

# Remove packages already installed
required_packages = required_packages[!required_packages %in% package]

# Install transformers
reticulate::py_install(c(
  required_packages
), pip = TRUE)



