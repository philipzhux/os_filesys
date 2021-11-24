#include "file_system.h"

// data input and output
__device__ __managed__ uchar input[MAX_FILE_SIZE_C];
__device__ __managed__ uchar output[MAX_FILE_SIZE_C];

// volume (disk storage)
__device__ __managed__ uchar volume[VOLUME_SIZE_C];


__device__ void user_program(FileSystem *fs, uchar *input, uchar *output);

__global__ void mykernel(uchar *input, uchar *output) {

  // Initilize the file system	
  FileSystem fs;
  printf("enter kernel \n");
  fs_init(&fs, volume, SUPERBLOCK_SIZE_C, FCB_SIZE_C, FCB_ENTRIES_C, 
			VOLUME_SIZE_C,STORAGE_BLOCK_SIZE_C, MAX_FILENAME_SIZE_C, 
			MAX_FILE_NUM_C, MAX_FILE_SIZE_C, FILE_BASE_ADDRESS_C);

  // user program the access pattern for testing file operations
  if(MAX_PER_DIR>MAX_FILE_SIZE_C/MAX_FILE_NUM_C)
  {
	printf("[ERROR] Max file# per dir can't exceed %d, check [config.h]\n",MAX_FILE_SIZE_C/MAX_FILE_NUM_C);
	assert(0);
  }
  if(MAX_PER_DIR>MAX_FILE_SIZE_C/MAX_FILE_NUM_C/sizeof(u16))
  printf(
	"[WARNING] Max file# per dir exceeding %d may lead to insufficient space for storage.\n",
	MAX_FILE_SIZE_C/MAX_FILE_NUM_C/sizeof(u16));
  user_program(&fs, input, output);
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "rb");

	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	//Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);
	return fileLen;
}

int main() {
  cudaError_t cudaStatus;
  load_binaryFile(DATAFILE, input, MAX_FILE_SIZE_C);
  cudaDeviceSetLimit(cudaLimitStackSize, 32768);
  // Launch to GPU kernel with single thread
  mykernel<<<1, 1>>>(input, output);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "mykernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return 0;
  }

  cudaDeviceSynchronize();
  cudaDeviceReset();

  write_binaryFile(OUTFILE, output, MAX_FILE_SIZE_C);


  return 0;
}
