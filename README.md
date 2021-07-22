# DLPC900

This is the repositroy for running the DLPC900 EVM DMD for the Thywissen Lab at the University of Toronto

The SDK, installer, and firmware may be found at Texas Instruments (free account required)

https://www.ti.com/product/DLPC900#design-development##software-development

This repository has the SDK and the firmware in the installation folder.  The GUI is too large to store on GitHub.

The DLPC900 EVM DMD has an automated kHz flicker to prevent mechnical decay of the mirrors.  We use as simple custom circuit to override this flickering when operating the DMD. One curious thing is that logic of the DMD appear to switch when the flickering is disabled.

