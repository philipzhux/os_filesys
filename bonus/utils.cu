__device__ int my_strcmp (const char * s1, const char * s2) 
{
    for(; *s1 == *s2; ++s1, ++s2)
        if(*s1 == 0)
            return 0;
    return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
}
  
  
__device__ char * my_strcpy(char *dest, const char *src)
{
    int i = 0;
    do {
    dest[i] = src[i];}
    while (src[i++] != 0);
    return dest;
}

__device__ char * my_strcat(char *dest, const char *src)
{
    int i = 0;
    while (dest[i] != 0) i++;
    my_strcpy(dest+i, src);
    return dest;
}



__device__ int my_strlen(const char *str_a)
{
    int len = 0;
    while(*str_a++) len++;
    return len;
}