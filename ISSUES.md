# Known Issues

In the following, known issues are listed.

## Faces detected on glasses
OFIQ uses the SSD face detector. SSD has a tendency to detect faces
on glasses as, for instance, when applied to the conformance test image
`r-06-pitch.png`. This affects the quality component `SingleFacePresent`.
To conform with the target value of `SingleFacePresent`, implementations 
conformant to the ISO/IEC 29794-5 standard shall detect the same erroneous
faces. Therefore, it is not possible to fix this in OFIQ v1.x.y.
