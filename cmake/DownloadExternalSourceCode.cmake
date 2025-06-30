set(FILE OFIQ-EXTERN.zip)
set(URL https://filex.secunet.com/?t=df5556766cc2418512b91ff7c7d58455)

message("Downloading external source code from ${URL}")
file(DOWNLOAD ${URL} ${FILE} SHOW_PROGRESS)
message("Extracting external source code")
file(ARCHIVE_EXTRACT 
    INPUT ${FILE}
    DESTINATION ../
    VERBOSE
)
file(REMOVE ${FILE})
