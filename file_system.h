#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2




struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};


__device__ void fs_init(FileSystem *fs, uchar *volum, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);
__device__ u32 find_best_fit(FileSystem *fs, int size);
__device__ void compact_disk(FileSystem *fs);
__device__ u32 offset_to_fd(FileSystem *fs, u32 offset);
__device__ inline unsigned char block_available(FileSystem *fs, u32 offset);
__device__ inline void set_bitmap(FileSystem *fs, u32 offset, u32 size,unsigned char t);
__device__ inline void set_bit(FileSystem *fs, u32 offset, unsigned char t);
__device__ u32 find_best_fit(FileSystem *fs, int size);
__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ inline void fs_delete_fd(FileSystem *fs,u32 fd);
__device__ inline u32 fs_insert_fcb(FileSystem *fs,char *s);
__device__ inline u32 fs_search(FileSystem *fs,char *s);
  __device__ int my_strcmp (const char * s1, const char * s2);
__device__ char * my_strcpy(char *dest, const char *src);
__device__ char * my_strcat(char *dest, const char *src);




#endif