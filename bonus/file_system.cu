#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#define REAL_SOTRAGE_BLOCKS 32768
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

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 curr_dir_fd = 0;

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
  // init bitmap
  set_bitmap(fs,0,MAX_FILE_SIZE,0);
  // init FCBs
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  for(u32 i=0; i<1024; i++) {
    SET_NAME(fcb_base,fd,"");
    SET_SIZE(fcb_base,fd,0);
    SET_DSIZE(fcb_base,fd,0);
    IS_DIR(fcb_base,fd) = 0
  }
  // init root
  init_root(fs);

}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  u32 fd = fs_search(fs,s);
	switch(op){
    case G_WRITE:
    if(fd>=1024) {
      //create file
      fd = fs_insert_fcb(fs,s);
      add_to_dir(fs,curr_dir_fd,fd);
    }
    
    if(fd>=1024) printf("[Error] Running out of space\n");
    break;

    case G_READ:
    if(fd>=1024) printf("[Error] Running out of space\n");
    break;
  }
  return fd;
}



__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
	u32 block_offset = GET_BLOCK_OFFSET(fcb_base,fd);
  if(size>GET_SIZE(fcb_base,fd)) size = GET_SIZE(fcb_base,fd);
  for(int i=0;i<size;i++) output[i] =
  fs->volume[fs->FILE_BASE_ADDRESS+block_offset*fs->STORAGE_BLOCK_SIZE+i];
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 old_size = GET_SIZE(fcb_base,fd);
	if(size!=old_size){
    set_bitmap(fs, GET_BLOCK_OFFSET(fcb_base,fd), old_size,0);
    /** insert fcb back in the right place **/
    u32 bf = find_best_fit(fs,size);
    if(bf>=REAL_SOTRAGE_BLOCKS){
      compact_disk(fs);
      bf = find_best_fit(fs,size);
      if(bf>=REAL_SOTRAGE_BLOCKS) printf("[ERROR] Running out of space\n");
    }
    set_bitmap(fs, bf, size,1);
    SET_BLOCK_OFFSET(fcb_base,fd,bf);
    SET_SIZE(fcb_base,fd,size);
  }
  SET_MTIME(fcb_base,fd,++gtime);
  u32 block_offset = GET_BLOCK_OFFSET(fcb_base,fd);
  for(int i=0;i<size;i++) fs->volume[fs->FILE_BASE_ADDRESS+block_offset*
  fs->STORAGE_BLOCK_SIZE+i] = input[i];
  return 0;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 count = 0;
  char name[1024][20];
  char is_dir[1024];
  int ctime[1024];
  ctime_new[count] = GET_CTIME(fcb_base,i);
  int mtime[1024];
  int size[1024];
  int index[1024];
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,curr_dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  fds_ptr++; //avoid intervening of the parent node
  while((*fds_ptr)!=1024){
    my_strcpy(name[count],(char*)NAME(fcb_base,(*fds_ptr)));
    ctime[count] = GET_CTIME(fcb_base,(*fds_ptr)); //reversed order with size
    mtime[count] = GET_MTIME(fcb_base,(*fds_ptr));
    size[count] = GET_SIZE(fcb_base,(*fds_ptr));
    if(IS_DIR(fcb_base,(*fds_ptr))) size[count] = GET_DSIZE(fcb_base,(*fds_ptr));
    index[count] = count;
    count++;
    fds_ptr++;
  }
  
	switch(op){
    case LS_D:
    thrust::sort_by_key(thrust::device, mtime, mtime + count, index, thrust::greater<int>());
    //modified time descending
    for(int i=0;i<count;i++){
      if(is_dir[index[i]]) {
        printf("%s\t%s\n",name[index[i]],"d");
      }
      else {
        printf("%s\n",name[index[i]]);
      }
    }

    break;

    case LS_S:
    printf("===sort by file size===\n");
    thrust::stable_sort_by_key(thrust::device, ctime, ctime + count, index);
    thrust::stable_sort_by_key(thrust::device, ctime_new, ctime_new + count, size);
    //create time accending
    thrust::stable_sort_by_key(thrust::device, size, size + count, index, thrust::greater<int>());
    //size descending
    for(int i=0;i<count;i++) {
      if(is_dir[index[i]]) {
        printf("%s\t%sB\t%s\n",name[index[i]],size[i],"d");
      }
      else{
        printf("%s\t%sB\n",name[index[i]],size[i]);
      }
    }
    break;

    case PWD:
    char buffer[1024];
    get_pwd(fs,buffer);
    print("%s\n",buffer);
    case CD_P:
    curr_dir_fd = get_parent_fd(fs,curr_dir_fd);
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  switch(op){
    case RM_RF:
    u32 fd  = fs_search(fs,s);
    if(fd>=1024)
    {
      printf("[ERROR] %s: No such file or directory\n",s);
      return;
    }
    rm_rf(fs,fd);
    break;
    case CD:
    u32 fd  = fs_search(fs,s);
    if(fd>=1024)
    {
      printf("[ERROR] %s: No such file or directory\n",s);
      return;
    }
    if(!IS_DIR(fcb_base,fd)){
      printf("[ERROR] %s: Not a directory\n",s);
      return;
    }
    curr_dir_fd = fd;
    break;
    case MKDIR:
    mk_dir(fs,s);
    break;
    default:
    break;
  }

}

