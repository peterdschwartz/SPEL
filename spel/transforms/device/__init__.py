"""
Device data layer for full-model OpenACC runs (`spel device-data`).

Generates `ElmDeviceDataMod.F90`: device initialization, a one-time deep copy of
every type instance to the device, and whole-state update routines; and inserts
the calls into `elm_initializeMod` and `elm_driver`. Design: DESIGN.md here.
"""
