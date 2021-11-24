#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H
#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <assert.h>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

#define NAME(fcb_base,fd) (fcb_base+fd*32)
#define SET_BLOCK_OFFSET(fcb_base,fd,pos) fcb_base[21+32*fd]=((pos&0x0000ff00)>>8); fcb_base[22+32*fd]=(pos&0x000000ff)
#define GET_BLOCK_OFFSET(fcb_base,fd) ((fcb_base[21+32*fd]<<8)|fcb_base[22+32*fd])

#define SET_CTIME(fcb_base,fd,time) fcb_base[23+32*fd]=((time&0x0000ff00)>>8);\
fcb_base[24+32*fd]=(time&0x000000ff)
#define GET_CTIME(fcb_base,fd) ((fcb_base[23+32*fd]<<8)|fcb_base[24+32*fd])

#define SET_MTIME(fcb_base,fd,time) fcb_base[25+32*fd]=((time&0x0000ff00)>>8);\
fcb_base[26+32*fd]=(time&0x000000ff)
#define GET_MTIME(fcb_base,fd) ((fcb_base[25+32*fd]<<8)|fcb_base[26+32*fd])
#define SET_SIZE(fcb_base,fd,size) fcb_base[27+32*fd]=((size&0x0000ff00)>>8);fcb_base[28+32*fd]=(size&0x000000ff)
#define GET_SIZE(fcb_base,fd) ((fcb_base[27+32*fd]<<8)|fcb_base[28+32*fd])
#define SET_DSIZE(fcb_base,fd,size) fcb_base[29+32*fd]=((size&0x0000ff00)>>8);fcb_base[30+32*fd]=(size&0x000000ff)
#define GET_DSIZE(fcb_base,fd) ((fcb_base[29+32*fd]<<8)|fcb_base[30+32*fd])
#define IS_DIR(fcb_base,fd) fcb_base[31+32*fd]

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

/** file_system.cu **/
__device__ void fs_init(FileSystem *fs, uchar *volum, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

/** api.cu **/
__device__ void compact_disk(FileSystem *fs);
__device__ u32 offset_to_fd(FileSystem *fs, u32 offset);
__device__ u32 find_best_fit(FileSystem *fs, int size);
__device__ void rm_rf(FileSystem *fs, u32 fd, u32 dir);
__device__ void fs_delete_fd(FileSystem *fs,u32 fd);
__device__ u32 fs_insert_fcb(FileSystem *fs,char *s);
__device__ u32 fs_search(FileSystem *fs,char *s);

/** bitmap.cu **/
__device__ unsigned char block_available(FileSystem *fs, u32 offset);
__device__ void set_bitmap(FileSystem *fs, u32 offset, u32 size,unsigned char t);
__device__ void set_bit(FileSystem *fs, u32 offset, unsigned char t);

/** dirs.cu **/
__device__ void init_root(FileSystem *fs);
__device__ void del_from_dir(FileSystem *fs, u16 dir_fd, u16 file_fd);
__device__ u32 mk_dir(FileSystem *fs,char *s);
__device__ u32 get_parent_fd(FileSystem *fs, u32 base_dir_fd);
__device__ void get_pwd(FileSystem *fs, char* buffer);
__device__ void add_to_dir(FileSystem *fs, u16 dir_fd, u16 file_fd);
__device__ u16 dir_count(FileSystem *fs, u16 dir_fd);

/** utils.cu **/
__device__ int my_strcmp (const char * s1, const char * s2);
__device__ char * my_strcpy(char *dest, const char *src);
__device__ char * my_strcat(char *dest, const char *src);
__device__ int my_strlen(const char *str_a);

/** global_var.cu **/
extern __device__ __managed__ u32 gtime;
extern __device__ __managed__ u32 curr_dir_fd;

#endif