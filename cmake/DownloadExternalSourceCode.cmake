set(FILE OFIQ-EXTERN.zip)

if (NOT DEFINED OFIQ_EXTERN_URL)
	set(OFIQ_EXTERN_URL https://standards.iso.org/iso-iec/29794/-5/ed-1/en/OFIQ-EXTERN.zip)
endif()

message("Downloading external source code from ${OFIQ_EXTERN_URL}")
file(DOWNLOAD ${OFIQ_EXTERN_URL} ${FILE} SHOW_PROGRESS)
message("Extracting external source code")
file(ARCHIVE_EXTRACT 
    INPUT ${FILE}
    DESTINATION ../
    VERBOSE
)
file(REMOVE ${FILE})
