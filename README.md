# Re-CoSKQ

## The Project Setup

## Running the Tests

## Building the Documentation
Requires make and sphinx to be installed.
First the .rst files need to be generated from the .py source files.
Once they are generated, the documentation can be built from the .rst files.
Possible targets include: html, singlehtml and latexpdf.

Use the following steps to generate the documentation:
 - Start from the root of the project (usually the re-coskq folder)
 - Generate the .rst files: sphinx-apidoc --implicit-namespaces -o docs/ src/
 - Go to the docs/ directory: cd docs/
 - (If something went wrong previously: make clean)
 - Build the target: make html