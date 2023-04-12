#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

void GetSemaphor(__global int * semaphor) {
  int occupied = atom_xchg(semaphor, 1);
  while(occupied > 0)
  {
   occupied = atom_xchg(semaphor, 1);
  }
}

void ReleaseSemaphor(__global int * semaphor)
{
  int prevVal = atom_xchg(semaphor, 0);
}

void Mul(float *, float *, float *, int, int, int, int);
void CopyVec(float *, float *, int);
float rand(float, float, float);

__kernel void
floatVectorSum(__global uchar4 * lastim,        // last image
__global uchar4 * newim,         // new image
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

    uchar4 color = (uchar4)(0, 0, 0, 255);
    
    if(evol[elw + 4] > 0 && evol[elw + 5] <= 0)     //if no energy => die
        evol[elw + 4] = 0;
    if(evol[elw + 4] > 0 && evol[elw + 5] <= 0)     //if die => destroy body
        evol[elw + 5] = 0;

    if(evol[elw + 4] > 1)
    {
        vec1[vecw + 0] = 0;        //8 neighbours
        vec1[vecw + 1] = 0;
        vec1[vecw + 2] = 0;
        vec1[vecw + 3] = 0;

        float red = 0;
        float green = 0;
        float blue = 0;
        
        for(int i = 7; i < 130; i+=1)
            red += evol[elw + i];
        for(int i = 131; i < 260; i+=1)
            green += evol[elw + i];
        for(int i = 261; i < 399; i+=1)
            blue += evol[elw + i];

        red = abs((int)red) * 15;
        green = abs((int)green) * 15;
        blue = abs((int)blue) * 15;

        color = (uchar4)(red, green, blue, 255);

        int f1 = elw - scrw * (int)size[3];
        int f2 = elw + (int)size[3];
        int f3 = elw + scrw * (int)size[3];
        int f4 = elw - (int)size[3];
        for(int i = 7; i < size[3]; i+=10)
        {
            vec1[vecw    ] += (evol[f1 + i] - evol[elw + i]);        //4 neighbours
            vec1[vecw + 1] += (evol[f2 + i] - evol[elw + i]);
            vec1[vecw + 2] += (evol[f3 + i] - evol[elw + i]);
            vec1[vecw + 3] += (evol[f4 + i] - evol[elw + i]);
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);

        vec1[vecw + 4] = evol[elw + 4] - 1;     //life time
        vec1[vecw + 5] = evol[elw + 5] - 1;     //energy
        vec1[vecw + 6] = 0;                    //position
        evol[elw + 4] = evol[elw + 4] - 1;
        evol[elw + 5] = evol[elw + 5] - 1;
        

        Mul(vec2, vec1, evol, 7, 7, id, 7 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 56 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 105 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 154 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 203 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 252 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 301 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 7, 7, id, 350 + elw);
        //CopyVec(vec1, vec2, vecw);
        //Mul(vec2, vec1, evol, 7, 7, id, 399 + elw);
        //CopyVec(vec1, vec2, vecw);
        //Mul(vec2, vec1, evol, 7, 5, id, 448 + elw);




        //barrier(CLK_GLOBAL_MEM_FENCE);

        float4 vv = (float4)(0, 0, 0, 0);
        vv.r = vec2[vecw + 3];
        vv.g = vec2[vecw + 4];
        vv.b = vec2[vecw + 5];
        vv.a = vec2[vecw + 6];

        int dir = elw - scrw * size[3];  
        //int dir = fmod((float)abs((int)(vec2[vecw + 4])), (float)4);

        vv =  normalize(vv);
        float max = vv.r;
        if(max < vv.g) {max = vv.g; dir = elw + size[3];; }
        if(max < vv.b) {max = vv.b; dir = elw + scrw * size[3];; }
        if(max < vv.a) {max = vv.a; dir = elw - size[3]; }

        int x = dir;

        /*
        if(dir == 0)        dir = elw - scrw * size[3];         //up
        else if(dir == 1)   dir = elw + size[3];                //right
        else if(dir == 2)   dir = elw + scrw * size[3];         //down
        else if(dir == 3)   dir = elw - size[3];                //left
        */

        int imdir = (dir - elw) / size[3];

        int action = 0;
        
        vv.r = vec2[vecw + 0];
        vv.g = vec2[vecw + 1];
        vv.b = vec2[vecw + 2];
        //vv.a = vec2[vecw + 3];

        vv =  normalize(vv);
        max = vv.r;
        if(max < vv.g) {max = vv.g; action = 1; }
        if(max < vv.b) {max = vv.b; action = 2; }
        //if(max < vv.a) {max = vv.a; action = 3; }

        if(imdir + id < scrw*2 || imdir + id > get_global_size(0) - scrw*2)
        {
            evol[elw + 4] = 0;
            evol[elw + 5] = 0;
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 0)                             //ATTACK
        {
            evol[dir + 4] = evol[elw + 4];
            evol[elw + 4] = 0;
            evol[dir + 5] = evol[elw + 5] + evol[dir + 5] - 0;
            evol[elw + 5] = 0;
            for(int i = 7; i < size[3]; i++)
            {
                evol[dir + i] = evol[elw + i];
            }
            
            //GENOCODE
            if(size[2] == 1)
            newim[id + imdir] = color;

            //GREEN OR RED
            if(size[2] == 0)
            {
                newim[id + imdir] = lastim[id];
                newim[id + imdir].r = newim[id + imdir].r + 4 < 255 ? newim[id + imdir].r + 4 : 255;
                newim[id + imdir].g = newim[id + imdir].g - 4 > 0 ? newim[id + imdir].g - 4 : 0;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 1)                    //PHOTOSYNTHESIS
        {
            evol[elw + 5] += 3;

            //GENOCODE
            if(size[2] == 1)
                newim[id] = color;

            //GREEN OR RED
            if(size[2] == 0)
            {
                newim[id] = lastim[id];
                newim[id].g = newim[id].g + 4 < 255 ? newim[id].g + 4 : 255;
                newim[id].r = newim[id].r - 4 > 0 ? newim[id].r - 4 : 0;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 2)                    //BORN
        {
            if(evol[dir + 4] == 0)
            {
                float rnd = rand(evol[elw + 4], evol[elw + 5], id);
                evol[dir + 5] = evol[elw + 5] / 2 - 0;
                evol[elw + 5] = evol[elw + 5] / 2 - 0;
                evol[dir + 4] = abs((int)(rnd * 20)) + 20;
                for(int i = 7; i < size[3]; i++)
                {
                    evol[dir + i] = evol[elw + i];
                }
                if(size[2] == 1)
                {
                    newim[id + imdir] = color;
                    newim[id] = color;
                }
                if(size[2] == 0)
                {
                    newim[id + imdir] = lastim[id];
                    newim[id] = lastim[id];
                }

                if(abs((int)rnd) < 0.2)
                {
                    for(int i = 0; i < 25; i++)
                    {
                        rnd = rand(evol[elw + 15], evol[elw + 10], i);
                        int s = abs((int)(rnd * (size[3]-7) + 7));
                        evol[dir + s] += rand(i, s, id) / 25;
                    }
                }
            }
            else
            {
                evol[elw + 5] -= 2;
                newim[id] = lastim[id];
            }
            
            
        }
    }
    else
        newim[id] = (uchar4)(0, 0, 0, 255);

}


void Mul(float * vec2, float * vec1, float * matrix, int width, int len, int index, int offset)
{
    int id = index * 7;
    for (unsigned int tx = 0; tx < len; tx++)
    {
        float value = 0;
        int txof = tx + offset;
        for (unsigned int k = 0; k < width; k++) {
            value += matrix[k * width + txof] * vec1[id + k];
        }
        vec2[id + tx] = value;
    }
}

void CopyVec(float * vec1, float * vec2, int id)
{
    for(unsigned int i = 0; i < 7; i++)
        vec1[id + i] = vec2[id + i];
}

float rand(float x, float y, float z) {
    //float ptr = 0.0f;
    //return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
    float rnd = cos(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f);
    return (rnd - (int)rnd);
}