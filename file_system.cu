﻿#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#define REAL_SOTRAGE_BLOCKS 32768
#define NAME(fcb_base,fd) (fcb_base+fd*32)
#define SET_BLOCK_OFFSET(fcb_base,fd,pos) fcb_base[21+32*fd]=((pos&0x00'00'ff'00)>>8); fcb_base[22+32*fd]=(pos&0x00'00'00'ff)
#define GET_BLOCK_OFFSET(fcb_base,fd) ((fcb_base[21+32*fd]<<8)|fcb_base[22+32*fd])

#define SET_CTIME(fcb_base,fd,time) fcb_base[23+32*fd]=((time&0x00'00'ff'00)>>8);\
fcb_base[24+32*fd]=(time&0x00'00'00'ff)
#define GET_CTIME(fcb_base,fd) ((fcb_base[23+32*fd]<<8)|fcb_base[24+32*fd])

#define SET_MTIME(fcb_base,fd,time) fcb_base[25+32*fd]=((time&0x00'00'ff'00)>>8);\
fcb_base[26+32*fd]=(time&0x00'00'00'ff)
#define GET_MTIME(fcb_base,fd) ((fcb_base[25+32*fd]<<8)|fcb_base[26+32*fd])
#define SET_SIZE(fcb_base,fd,size) fcb_base[27+32*fd]=((size&0x00'00'ff'00)>>8);fcb_base[28+32*fd]=(size&0x00'00'00'ff)
#define GET_SIZE(fcb_base,fd) ((fcb_base[27+32*fd]<<8)|fcb_base[28+32*fd])

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}

__device__ void compact_disk(FileSystem *fs)
{
  u32 fd;
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 vacant_front,chunk_front = 0;
  while(chunk_front<REAL_SOTRAGE_BLOCKS) {
    while(!block_available(fs,vacant_front) && vacant_front<REAL_SOTRAGE_BLOCKS) vacant_front++;
    chunk_front = vacant_front;
    while(chunk_front<REAL_SOTRAGE_BLOCKS && block_available(fs,chunk_front)) chunk_front++;
    if(chunk_front<REAL_SOTRAGE_BLOCKS && (fd = offset_to_fd(fs,chunk_front))<1024){
      int size = GET_SIZE(fcb_base,fd);
      for(int i=0;i<size;i++) fs->volume[fs->FILE_BASE_ADDRESS+vacant_front*
      fs->STORAGE_BLOCK_SIZE+i] = fs->volume[fs->FILE_BASE_ADDRESS+chunk_front*
      fs->STORAGE_BLOCK_SIZE+i];
      set_bitmap(fs,chunk_front,size,0);
      set_bitmap(fs,vacant_front,size,1);
      SET_BLOCK_OFFSET(fcb_base,fd,vacant_front);
      vacant_front += (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
    }
  }
}

__device__ u32 offset_to_fd(FileSystem *fs, u32 offset) {
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 fd;
  for(fd=0;fd<1024;fd++){
    if(GET_BLOCK_OFFSET(fcb_base,fd)==offset) break;
  }
  return fd;
}

__device__ inline unsigned char block_available(FileSystem *fs, u32 offset)
{
  if(offset>(1<<15)) return 0;
  u32 map_byte = offset/8;
  u32 map_bit = offset%8;
  return (((fs->volume[map_byte]>>map_bit) & 1)==0);
}
__device__ inline void set_bitmap(FileSystem *fs, u32 offset, u32 size,unsigned char t) {
  size = (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
  for(u32 i=offset;i<offset+size;i++) set_bit(fs,i,t);
}

__device__ inline void set_bit(FileSystem *fs, u32 offset, unsigned char t) {
  u32 map_byte = offset/8;
  u32 map_bit = offset%8;
  if(t){
    fs->volume[map_byte] = fs->volume[map_byte] | (0x1<<map_bit);
    return;
  }
  fs->volume[map_byte] = fs->volume[map_byte] & ~(0x1<<map_bit);
  return;
}

__device__ BestFit find_best_fit(FileSystem *fs, int size)
{
  int best_fit = REAL_SOTRAGE_BLOCKS;
  int bf_blocks = REAL_SOTRAGE_BLOCKS;
  int block_needed = (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
  int block = 0;
  int chunk = 0;
  int cursor = 0;
  while(block<REAL_SOTRAGE_BLOCKS)
  {
    if(!block_available(fs,block))
    {
      while(!block_available(fs,block)) block++;
      chunk = 0;
      cursor = block;
      continue;
    }
    if(chunk>=block_needed && chunk<bf_blocks)
    {
      best_fit = cursor;
      bf_blocks = chunk;
    }
    block++;
    chunk++;
  }
  return best_fit;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  u32 fd = fs_search(fs,char);
	switch(op){
    case G_WRITE:
    if(fd>=1024) fd = fs_insert_fcb(fs,s);
    if(fd>=1024) printf("[Error] Running out of space\n");
    break;

    case G_READ:
    if(fd>=1024) printf("[Error] Running out of space\n");
    return fd;
    break;
  }
}

__device__ inline u32 fs_delete_fd(FileSystem *fs,u32 fd)
{
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  set_bitmap(fs, GET_BLOCK_OFFSET(fcb_base,fd), GET_SIZE(fcb_base,fd),0);
  SET_NAME(fcb_base,fd,"");
  SET_SIZE(fcb_base,fd,0);
}

__device__ inline u32 fs_insert_fcb(FileSystem *fs,char *s)
{
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  for(u32 i=0; i<1024; i++) {
    if(my_strcmp(GET_NAME(fcb_base,i),"")){
      u32 ts = (u32)time(NULL);
      SET_NAME(fcb_base,i,s);
      SET_SIZE(fcb_base,i,0);
      SET_CTIME(fs,fd,++gtime);
      return i;
    }
  }
  return 1024;
}
__device__ inline u32 fs_search(FileSystem *fs,char *s)
{
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 curr = 0;
  while(!my_strcmp(GET_NAME(fcb_base,curr),s) && curr<1024) curr++;
  return curr;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fd)
{
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
	u32 block_offset = GET_BLOCK_OFFSET(fcb_base,fd);
  if(size>GET_SIZE(fcb_base,fd)) size = GET_SIZE(fcb_base,fd);
  for(int i=0;i<size;i++) output[i] =
  fs->volume[fs->FILE_BASE_ADDRESS+block_offset*fs->STORAGE_BLOCK_SIZE+i];
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fd)
{
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 old_size = GET_SIZE(fcb_base,fd);
	if(size!=old_size){
    set_bitmap(fs, GET_BLOCK_OFFSET(fcb_base,fd), old_size,0);
    /** insert fcb back in the right place **/
    u32 bf = find_best_fit(fs,size);
    if(bf>=REAL_SOTRAGE_BLOCKS){
      compact_disk(fs);
      if((bf = find_best_fit(fs,size)>=REAL_SOTRAGE_BLOCKS) printf("[ERROR] Running out of space\n");
    }
    set_bitmap(fs, bf, size,1);
    SET_BLOCK_OFFSET(fcb_base,fd,bf.best_fit_block);
    SET_SIZE(fcb_base,fd,size);
  }
  SET_MTIME(fs,fd,++gtime);
  u32 block_offset = GET_BLOCK_OFFSET(fcb_base,fd);
  for(int i=0;i<size;i++) fs->volume[fs->FILE_BASE_ADDRESS+block_offset*
  fs->STORAGE_BLOCK_SIZE+i] = input[i];
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 count = 0;
  char name[1024][20];
  int ctime[1024];
  int mtime[1024];
  int size[1024];
  int index[1024];
  for(u32 i=0; i<1024; i++) {
    if(my_strcmp(GET_NAME(fcb_base,i),"")){
      continue;
    }
    my_strcpy(name[count],GET_NAME(fcb_base,i));
    ctime[count] = GET_CTIME(fcb_base,i); //reversed order with size
    mtime[count] = GET_MTIME(fcb_base,i);
    size[count] = GET_SIZE(fcb_base,i);
    index[count] = count;
    count++;
  }
	switch(op){
    case LS_D:
    thrust::sort_by_key(thrust::device, mtime, mtime + count, index, thrust::greater<int>());
    //modified time descending
    for(int i=0;i<count;i++)
      printf("%s\n",name[index[i]]);
    break;

    case LS_S:
    thrust::sort_by_key(thrust::device, ctime, ctime + count, index);
    //create time accending
    thrust::stable_sort_by_key(thrust::device, size, size + count, index, thrust::greater<int>());
    //size descending
    for(int i=0;i<count;i++)
      printf("%s\t%sB\n",name[index[i]],size[index[i]]);
    break;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  u32 fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 fd  = fs_search(fs,s);
  fs_delete_fd(fs,fd);

}

__device__ int my_strcmp(const char *str_a, const char *str_b, unsigned len = 20){
  int match = 0;
  unsigned i = 0;
  unsigned done = 0;
  while ((i < len) && (match == 0) && !done){
    if ((str_a[i] == 0) || (str_b[i] == 0)) done = 1;
    else if (str_a[i] != str_b[i]){
      match = i+1;
      if ((int)str_a[i] - (int)str_b[i]) < 0) match = 0 - (i + 1);}
    i++;}
  return match;
  }



  __device__ char * my_strcpy(char *dest, const char *src){
    int i = 0;
    do {
      dest[i] = src[i];}
    while (src[i++] != 0);
    return dest;
  }
  
  __device__ char * my_strcat(char *dest, const char *src){
    int i = 0;
    while (dest[i] != 0) i++;
    my_strcpy(dest+i, src);
    return dest;
  }