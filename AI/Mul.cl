void Mulv(float *, float *, float *, int, int, int, int);
void CopyVec(float *, float *, int);

__kernel void
floatVectorSum(
__global float * vec1,          // first vector for matrix mul and its result
__global float * vec2,          // second vector for matrix mul
__global float * evol,          // neural net coefficients
                                // 1x7 [11] - in; 
                                // 7x7 [49] - hiden layer
                                // 5x7 [35] - out
__global float * size           // support nums ([screen width], [vector width], [vision], [matrix width])
)
{
    int id = get_global_id(0);   //id of pixel
    int elw = id * size[3];      //evol width
    int vecw = id * size[1];     //vec width
    int scrw = size[0];          //screen width
    

    Mulv(vec2, vec1, evol, 2, 2, id, 2 + elw);  // 3 <= 2
    //CopyVec(vec1, vec2, vecw);
    //Mulv(vec2, vec1, evol, 3, 3, id, 8 + elw);
    ///CopyVec(vec1, vec2, vecw);
    //Mulv(vec2, vec1, evol, 2, 3, id, 17 + elw);

}



void Mulv(float * vec2, float * vec1, float * matrix, int width, int len, int index, int offset)
{
    int id = index * len;
    for (int tx = 0; tx <= len; tx++)
    {
        float value = 0;
        for (int k = 0; k <= width; ++k) {
            value += matrix[k * width + tx + offset] * vec1[id + k];
        }
        vec2[id + tx] = value + matrix[offset + width * len];
    }
}

void CopyVec(float * vec1, float * vec2, int id)
{
    for(unsigned int i = 0; i < 3; i++)
        vec1[id + i] = vec2[id + i];
}