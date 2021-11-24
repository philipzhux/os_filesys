#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define MAX_PER_DIR 256
#define REAL_SOTRAGE_BLOCKS 32768

#define SUPERBLOCK_SIZE_C 4096 //32K/8 bits = 4 K
#define FCB_SIZE_C 32 //32 bytes per FCB
#define FCB_ENTRIES_C 1024
#define VOLUME_SIZE_C 1085440 //4096+32768+1048576
#define STORAGE_BLOCK_SIZE_C 32

#define MAX_FILENAME_SIZE_C 20
#define MAX_FILE_NUM_C 1024
#define MAX_FILE_SIZE_C 1048576

#define FILE_BASE_ADDRESS_C 36864 //4096+32768

#define DIR_END FCB_ENTRIES_C
#define FD_END FCB_ENTRIES_C

#define G_WRITE 0
#define G_READ 1
#define LS_D 2
#define LS_S 3
#define RM 4
#define CD 5
#define MKDIR 6
#define PWD 7
#define CD_P 8
#define RM_RF 9