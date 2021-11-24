#include "file_system.h"

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
  for(u32 i=0; i<FD_END; i++) {
    NAME(fcb_base,i)[0] = '\0';
    SET_SIZE(fcb_base,i,0);
    SET_DSIZE(fcb_base,i,0);
    IS_DIR(fcb_base,i) = 0;
  }
  // init root
  init_root(fs);

}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  u32 fd = fs_search(fs,s);
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  if(fd<FD_END && IS_DIR(fcb_base,fd))
  {
    printf("[ERROR] Can't open directory %s.\n",s);
    printf("The FD is %d\n",fd);
    assert(0);
    return FD_END;
  }
	switch(op){
    case G_WRITE:
    if(fd>=FD_END) {
      //create file
      if(dir_count(fs,curr_dir_fd)>MAX_PER_DIR-1){
        printf("Files and subdirs at each directory cannot exceed %d.\n",MAX_PER_DIR);
        printf("The files in current path FYI:\n");
        fs_gsys(fs, LS_S);
        printf("Again, files and subdirs at each directory cannot exceed %d.\n",MAX_PER_DIR);
        assert(0);
      }
      fd = fs_insert_fcb(fs,s);
      add_to_dir(fs,curr_dir_fd,fd);
    }
    
    if(fd>=FD_END) printf("[ERROR] Running out of space\n");
    assert(fd<FD_END);
    break;

    case G_READ:
    if(fd>=FD_END) printf("[ERROR] Running out of space\n");
    assert(fd<FD_END);
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
      assert(bf<REAL_SOTRAGE_BLOCKS);
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
  u16* ctime = new u16[MAX_PER_DIR];
  u16* ctime_new = new u16[MAX_PER_DIR];
  u16* mtime = new u16[MAX_PER_DIR];
  u16* size = new u16[MAX_PER_DIR];
  u16* index = new u16[MAX_PER_DIR];
  u16* fds = new u16[MAX_PER_DIR];
  // char** name = new char*[50];
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,curr_dir_fd);
  u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  fds_ptr++; //avoid intervening of the parent node
  while((*fds_ptr)!=DIR_END){
    u32 rfd = *fds_ptr;
    // name[count]=(char*)NAME(fcb_base,rfd);
    ctime[count] = GET_CTIME(fcb_base,rfd);
    ctime_new[count] = GET_CTIME(fcb_base,rfd);
    mtime[count] = GET_MTIME(fcb_base,rfd);
    size[count] = GET_SIZE(fcb_base,rfd);
    //is_dir[count] = IS_DIR(fcb_base,rfd);
    if(IS_DIR(fcb_base,rfd)) size[count] = GET_DSIZE(fcb_base,rfd);
    index[count] = count;
    fds[count] = rfd;
    count++;
    fds_ptr++;
  }
  
	switch(op){
    case LS_D:
    {
      printf("===sort by modified time===\n");
      thrust::sort_by_key(thrust::device, mtime, mtime + count, index, thrust::greater<u16>());
      //modified time descending
      for(int i=0;i<count;i++){
        if(IS_DIR(fcb_base,fds[index[i]])) {
          printf("%s\t%s\n",NAME(fcb_base,fds[index[i]]),"d");
        }
        else {
          printf("%s\n",NAME(fcb_base,fds[index[i]]));
        }
      }
  
      break;
    }
    case LS_S:
    {
      printf("===sort by file size===\n");
      thrust::stable_sort_by_key(thrust::device, ctime, ctime + count, index);
      thrust::stable_sort_by_key(thrust::device, ctime_new, ctime_new + count, size);
      //create time accending
      thrust::stable_sort_by_key(thrust::device, size, size + count, index, thrust::greater<u16>());
      //size descending
      for(int i=0;i<count;i++) {
        if(IS_DIR(fcb_base,fds[index[i]])) {
          printf("%s\t%d\t%s\n",NAME(fcb_base,fds[index[i]]),size[i],"d");
        }
        else{
          printf("%s\t%d\n",NAME(fcb_base,fds[index[i]]),size[i]);
        }
      }
      break;
    }
    case PWD:
    {
      char* buffer = new char[4098];
      get_pwd(fs,buffer);
      printf("%s\n",buffer);
      free(buffer);
      break;
    }
    case CD_P:
    {
      curr_dir_fd = get_parent_fd(fs,curr_dir_fd);
      //printf("[CD_P] Going to enclosing directory...\n");
      break;
    }
    default:
    break;
  }
  free(ctime);
  free(ctime_new);
  free(mtime);
  free(size);
  free(index);
  free(fds);
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 fd;
  switch(op){
    case RM_RF:
    {
      fd  = fs_search(fs,s);
      if(fd>=FD_END)
      {
        printf("[ERROR] %s: No such file or directory\n",s);
        assert(0);
        return;
      }
      rm_rf(fs,fd,curr_dir_fd);
      //printf("[RM_RF] %s already recursively deleted...\n",s);
      break;
    }
    case CD:
    {
      fd  = fs_search(fs,s);
      if(fd>=FD_END)
      {
        printf("[ERROR] %s: No such file or directory\n",s);
        assert(0);
        return;
      }
      if(!IS_DIR(fcb_base,fd)){
        printf("[ERROR] %s: Not a directory\n",s);
        assert(0);
        return;
      }
      curr_dir_fd = fd;
      //printf("[CD] Jumping to subdir: %s...\n",s);
      break;
    }
    case MKDIR:
    {
      u32 fd  = fs_search(fs,s);
      if(fd<FD_END) {
        printf("[ERROR] MKDIR: %s already exists in current dir\n",s);
        assert(0);
        return;
      }
      if(dir_count(fs,curr_dir_fd)>=MAX_PER_DIR){
        printf("Files and subdirs at each directory cannot exceed %d.\n",MAX_PER_DIR);
        printf("The files in current path FYI:\n");
        fs_gsys(fs, LS_S);
        printf("Again, files and subdirs at each directory cannot exceed %d.\n",MAX_PER_DIR);
        assert(0);
      }
      mk_dir(fs,s);
    }
    break;
    default:
    break;
  }

}

