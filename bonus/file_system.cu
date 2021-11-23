#include "file_system.h"
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
#define SET_DSIZE(fcb_base,fd,size) fcb_base[29+32*fd]=((size&0x00'00'ff'00)>>8);fcb_base[30+32*fd]=(size&0x00'00'00'ff)
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
  set_bitmap(fs,0,MAX_FILE_SIZE,0); //set all available
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  for(u32 i=0; i<1024; i++) {
    SET_NAME(fcb_base,fd,"");
    SET_SIZE(fcb_base,fd,0);
  } //initialize all fcbs

}

__device__ void init_root(FileSystem *fs)
{
  /** init root dir **/
}

__device__ void rm_rf(FileSystem *fs, u32 fd){
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  if(!IS_DIR(fcb_base,fd)){
    fs_delete_fd(fs,fd);
    del_from_dir(fs,curr_dir_fd,fd);
    return;
  }
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,base_dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  fds_ptr++; //avoid deleting the parent dir
  while((*fds_ptr)!=1024) {
    rm_rf(fs,fd);
    fds_ptr++;
  }
  /* finish deleting subdir and files, delete self */
  fs_delete_fd(fs,fd);
  del_from_dir(fs,curr_dir_fd,fd);
}

__device__ u32 mk_dir(FileSystem *fs,char *s);
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 new_fd = fs_insert_fcb(FileSystem *fs,char *s);
  SET_SIZE(fcb_base,fd，1024);
  IS_DIR(fcb_base,fd) = 1;
  u32 new_dir_block_offset = GET_BLOCK_OFFSET(fcb_base,new_fd);
  u16* new_fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  (*new_fds_ptr++) = curr_dir_fd; //add curr_dir as the parent of the new dir
  *new_fds_ptr = 1024; //add end
  add_to_dir(FileSystem *fs, curr_dir_fd, new_fd); //add new dir to curr_dir
  return new_fd;
}

__device__ u32 get_parent_fd(FileSystem *fs, u32 base_dir_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,base_dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  u32 parent = *fds_ptr;
  return parent;
}


__device__ void get_pwd(FileSystem *fs, char* buffer)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  char names[5][20];
  u32 dir_fd = curr_dir_fd;
  int count = 0;
  while(dir_fd)
  {
    my_strcpy(names[count++],GET_NAME(fcb_base,dir_fd));
    dir_fd = get_parent_fd(fs,dir_fd);
  }
  my_strcpy(buffer,"/");
  for(int t=count-1;i>=0;t--){
    my_strcmp(buffer,names[t]);
  }
}


__device__ void add_to_dir(FileSystem *fs, u16 dir_fd, u16 file_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  while((*fds_ptr)!=1024) fds_ptr++;
  *(fds_ptr++)=file_fd;
  *(fds_ptr)=1024;
  SET_DSIZE(fcb_base,dir_fd,GET_DSIZE(fcb_base,dir_fd)+my_strlen(GET_NAME(fcb_base,file_fd))+1);
}


__device__ void del_from_dir(FileSystem *fs, u16 dir_fd, u16 file_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  while((*fds_ptr)!=file_fd) fds_ptr++;
  u16* tar_ptr = fds_ptr;
  while((*fds_ptr)!=1024) fds_ptr++;
  /** copy last to target **/
  *tar_ptr = *(--fds_ptr);
  /** delete last **/
  *(fds_ptr) = 1024;
  SET_DSIZE(fcb_base,dir_fd,GET_DSIZE(fcb_base,dir_fd)-my_strlen(GET_NAME(fcb_base,file_fd))-1);
}

__device__ void compact_disk(FileSystem *fs)
{
  u32 fd;
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
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
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
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
  u32 block_size = (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
  for(u32 i=offset;i<offset+block_size;i++) set_bit(fs,i,t);
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
    if(fd>=1024) {
      //create file
      fd = fs_insert_fcb(fs,s);
      add_to_dir(fs,curr_dir_fd,fd);
    }
    
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
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  set_bitmap(fs, GET_BLOCK_OFFSET(fcb_base,fd), GET_SIZE(fcb_base,fd),0);
  SET_NAME(fcb_base,fd,"");
  SET_SIZE(fcb_base,fd,0);
}

__device__ inline u32 fs_insert_fcb(FileSystem *fs,char *s)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  for(u32 i=0; i<1024; i++) {
    if(my_strcmp(GET_NAME(fcb_base,i),"")){
      u32 ts = (u32)time(NULL);
      SET_NAME(fcb_base,i,s);
      SET_SIZE(fcb_base,i,0);
      SET_CTIME(fs,fd,++gtime);
      SET_DSIZE(fcb_base,fd，0);
      IS_DIR(fcb_base,fd) = 0;
      return i;
    }
  }
  return 1024;
}
__device__ inline u32 fs_search(FileSystem *fs,char *s)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,curr_dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  fds_ptr++; //avoid intervening of the parent node
  while((*fds_ptr)!=1024){
    if(my_strcmp(GET_NAME(fcb_base,*fds_ptr),s)) return (*fds_ptr);
    fds_ptr++;
  }
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
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 count = 0;
  char name[1024][20];
  char is_dir[1024];
  int ctime[1024];
  int mtime[1024];
  int size[1024];
  int index[1024];
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,curr_dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  fds_ptr++; //avoid intervening of the parent node
  while((*fds_ptr)!=1024){
    my_strcpy(name[count],GET_NAME(fcb_base,(*fds_ptr)));
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
    thrust::sort_by_key(thrust::device, ctime, ctime + count, index);
    //create time accending
    thrust::stable_sort_by_key(thrust::device, size, size + count, index, thrust::greater<int>());
    //size descending
    for(int i=0;i<count;i++) {
      if(is_dir[index[i]]) {
        printf("%s\t%sB\t%s\n",name[index[i]],size[index[i]],"d");
      }
      else{
        printf("%s\t%sB\n",name[index[i]],size[index[i]]);
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



  __device__ int my_strlen(const char *str_a){
    int len = 0;
    while(*str_a++) len++;
    return len;
    }