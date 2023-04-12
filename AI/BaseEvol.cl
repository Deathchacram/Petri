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
__global float * evol,          // neural net coefficients. 308 for each ([11], [121], [121], [55])
                                // 1x11 [11] - in; 
                                // 11x11 [121] - hiden layer 1; 
                                // 11x11 [121] - hiden layer 2; 
                                // 5x11 [55] - out
__global float * size           // support nums ([screen width], [matrix width])
)
{
    int id = get_global_id(0);   //id of pixel
    int elw = id * size[3];          //evol width
    int vecw = id * 11;          //evol width
    int scrw = size[0];          //screen width
    ulong counter = 100;
    uchar4 color = (uchar4)(0, 0, 0, 255);
    
    if(evol[elw + 8] > 0 && evol[elw + 9] <= 0)     //if no energy => die
        evol[elw + 8] = 0;
    if(evol[elw + 8] > 0 && evol[elw + 9] <= 0)     //if die => destroy body
        evol[elw + 9] = 0;

    if(evol[elw + 8] > 1)
    {
        vec1[vecw + 0] = 0;        //8 neighbours
        vec1[vecw + 1] = 0;
        vec1[vecw + 2] = 0;
        vec1[vecw + 3] = 0;
        vec1[vecw + 4] = 0;
        vec1[vecw + 5] = 0;
        vec1[vecw + 6] = 0;
        vec1[vecw + 7] = 0;

        float red = 0;
        float green = 0;
        float blue = 0;
        
        /*for(int i = 11; i < 105; i+=1)
            red = red + abs((int)(evol[elw + i] / 25));
        for(int i = 106; i < 207; i+=1)
            green = green + abs((int)(evol[elw + i] / 25));
        for(int i = 208; i < 308; i+=1)
            blue = blue +  abs((int)(evol[elw + i] / 25));

        for(int i = 11; i < 105; i+=1)
            red += evol[elw + i];
        for(int i = 106; i < 207; i+=1)
            green += evol[elw + i];
        for(int i = 208; i < 308; i+=1)
            blue += evol[elw + i];

        red = abs((int)red);
        green = abs((int)green);
        blue = abs((int)blue);
        color = (uchar4)(red, green, blue, 255);*/

        for(int i = 11; i < size[3]; i+=10)
        {
            vec1[vecw + 0] += (evol[elw - scrw * (int)size[3] + i]                  - evol[elw + i])    / (size[3] - 11);        //8 neighbours
            vec1[vecw + 1] += (evol[elw - scrw * (int)size[3] + (int)size[3] + i]   - evol[elw + i])    / (size[3] - 11);
            vec1[vecw + 2] += (evol[elw + (int)size[3] + i]                         - evol[elw + i])    / (size[3] - 11);
            vec1[vecw + 3] += (evol[elw + scrw * (int)size[3] + (int)size[3] + i]   - evol[elw + i])    / (size[3] - 11);
            vec1[vecw + 4] += (evol[elw + scrw * (int)size[3] + i]                  - evol[elw + i])    / (size[3] - 11);
            vec1[vecw + 5] += (evol[elw + scrw * (int)size[3] - (int)size[3] + i]   - evol[elw + i])    / (size[3] - 11);
            vec1[vecw + 6] += (evol[elw - (int)size[3] + i]                         - evol[elw + i])    / (size[3] - 11);
            vec1[vecw + 7] += (evol[elw - scrw * (int)size[3] - (int)size[3] + i]   - evol[elw + i])    / (size[3] - 11);
        }

        /*vec1[vecw + 0] = 1 + rand(evol[elw + 8 - scrw * size[3]], counter++, counter++);        //8 neighbours
        vec1[vecw + 1] = 2 + rand(evol[elw + 8 - scrw * size[3] + size[3]], counter++, counter++);
        vec1[vecw + 2] = 3 + rand(evol[elw + 8 + size[3]], counter++, counter++);
        vec1[vecw + 3] = 4 + rand(evol[elw + 8 + scrw * size[3] + size[3]], counter++, counter++);
        vec1[vecw + 4] = 5 + rand(evol[elw + 8 + scrw * size[3]], counter++, counter++);
        vec1[vecw + 5] = 6 + rand(evol[elw + 8 + scrw * size[3] - size[3]], counter++, counter++);
        vec1[vecw + 6] = 7 + rand(evol[elw + 8 - size[3]], counter++, counter++);
        vec1[vecw + 7] = 8 + rand(evol[elw + 8 - scrw * size[3] - size[3]], counter++, counter++);*/
        
        barrier(CLK_GLOBAL_MEM_FENCE);

        vec1[vecw + 8] = evol[elw + 8] - 1;     //life time
        vec1[vecw + 9] = evol[elw + 9] - 1;     //energy
        vec1[vecw + 10]= id;                    //position
        evol[elw + 8] = evol[elw + 8] - 1;
        evol[elw + 9] = evol[elw + 9] - 1;
        
        float curL = vec1[vecw + 8] + vec1[vecw + 9];

        Mul(vec2, vec1, evol, 11, 11, id, 11 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 11, 11, id, 132 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 11, 11, id, 253 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 11, 11, id, 374 + elw);
        CopyVec(vec1, vec2, vecw);
        Mul(vec2, vec1, evol, 11, 5, id, 495 + elw);




        //barrier(CLK_GLOBAL_MEM_FENCE);

        float4 vv = (float4)(0, 0, 0, 0);
        vv.r = vec2[vecw + 0];
        vv.g = vec2[vecw + 1];
        vv.b = vec2[vecw + 2];
        vv.a = vec2[vecw + 3];

        //int dir = sin((vec2[vecw + 4]*vec2[vecw + 4] - vec2[vecw + 5]*vec2[vecw + 5])) * 2 + 2;
        //int dir = ((vec2[4] - 8.0 * (vec2[3] / 8.0)));

        int dir = fmod((float)abs((int)(vec2[vecw + 4])), (float)8);
        //int dir = 0;
        
        int x = dir;

        if(dir == 0)        dir = elw - scrw * size[3];         //up
        else if(dir == 1)   dir = elw - scrw * size[3] + size[3];   //up-right
        else if(dir == 2)   dir = elw + size[3];                //right
        else if(dir == 3)   dir = elw + scrw * size[3] + size[3];   //right-down
        else if(dir == 4)   dir = elw + scrw * size[3];         //down
        else if(dir == 5)   dir = elw + scrw * size[3] - size[3];   //down-left
        else if(dir == 6)   dir = elw - size[3];                //left
        else if(dir == 7)   dir = elw - scrw * size[3] - size[3];   //left-up
        
        int imdir = (dir - elw) / size[3];

        int action = 0;

        vv =  normalize(vv);
        float max = vv.r;
        if(max < vv.g) {max = vv.g; action = 1; }
        if(max < vv.b) {max = vv.b; action = 2; }
        if(max < vv.a) {max = vv.a; action = 3; }

        if(curL == evol[elw + 8] + evol[elw + 9])
        {
        if(action == 0)                    //ATTACK
        {
            evol[dir + 8] = evol[elw + 8];
            evol[elw + 8] = 0;
            evol[dir + 9] = evol[elw + 9] + evol[dir + 9] - 1;
            evol[elw + 9] = 0;
            for(int i = 11; i < size[3]; i++)
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
                newim[id + imdir].r = newim[id + imdir].r + 8 < 255 ? newim[id + imdir].r + 8 : 255;
                newim[id + imdir].g = newim[id + imdir].g - 8 > 0 ? newim[id + imdir].g - 8 : 0;
            }
            //ENERGY
            //newim[id + imdir] = (uchar4)(0, 0, 0, 255);
            
            //newim[id] = (uchar4)(0, 0, 0, 255);
            //lastim[id] = (uchar4)(0, 0, 0, 255);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 1)                    //PHOTOSYNTHESIS
        {
            evol[elw + 9] += 6;

            //GENOCODE
            if(size[2] == 1)
                newim[id] = color;

            //GREEN OR RED
            if(size[2] == 0)
            {
                newim[id] = lastim[id];
                newim[id].g = newim[id].g + 8 < 255 ? newim[id].g + 8 : 255;
                newim[id].r = newim[id].r - 8 > 0 ? newim[id].r - 8 : 0;
            }

            //ENERGY
            //newim[id] = (uchar4)(0, 0, 0, 255);
            //newim[id].g = evol[elw + 9] * 2 < 255 ? evol[dir + 9] * 10 : 255;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 2)                             //MOVE
        {
            int sht = -2;
            if(evol[dir + 8] == 0)
            {
                evol[dir + 8] = evol[elw + 8];
                evol[elw + 8] = 0;
                evol[dir + 9] = evol[elw + 9] - sht;
                evol[elw + 9] = 0;
                for(int i = 11; i < size[3]; i++)
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
                    //newim[id + imdir] = (uchar4)(127, 127, 0, 255);
                    //newim[id] = (uchar4)(127, 127, 127, 255);
                    //newim[id + imdir].b = newim[id].b + 10 < 255 ? newim[id].b + 10 : 255;
                }
                
                //ENERGY
                //newim[id + imdir] = (uchar4)(0, 0, 0, 255);
                //newim[id + imdir].g = evol[dir + 9] * 2 < 255 ? evol[dir + 9] * 10 : 255;


                //newim[id + imdir] = (uchar4)(0, evol[dir + 9] * 2, 0, 255);
                
                //newim[id] = lastim[id];
                //newim[id].g = newim[id].g + 2 < 255 ? newim[id].g + 2 : 255;

                //newim[id] = (uchar4)(0, 0, 0, 255);
                lastim[id] = (uchar4)(0, 0, 0, 255);

                //newim[id + imdir] = (uchar4)(0, 255, 0, 255 );
                //newim[id + imdir].r += 10;
                //newim[id - scrw] = (uchar4)(0, evol[dir + 8] * 5, 0, 255);
                //newim[id - scrw] = (uchar4)(0, fmod((float)abs((int)(vec2[vecw + 4])), (float)4) * 50 + 40, 0, 255);
            }
            else
            {
                evol[elw + 9] -= sht;
                newim[id] = lastim[id];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 3)                    //BORN
        {
            if(evol[dir + 8] == 0)
            {
                float rnd = rand(evol[elw + 8], evol[elw + 9], id);
                evol[dir + 9] = evol[elw + 8] / 2;
                evol[elw + 9] = evol[elw + 9] / 2;
                evol[dir + 8] = abs((int)(rnd * 10)) + 10;
                for(int i = 11; i < size[3]; i++)
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
                    //newim[id + imdir] = (uchar4)(127, 127, 0, 255);
                    //newim[id] = (uchar4)(127, 127, 0, 255);
                    
                    newim[id + imdir] = lastim[id];
                    newim[id] = lastim[id];
                }

                if(abs((int)rnd) < 0.2)
                {
                    int s = abs((int)(rnd * (size[3]-11) + 11));
                    for(int i = s; i < s + 10; i++)
                        evol[dir + i] += rand(i, s, id) / 10;
                    //newim[id + imdir].b = newim[id].b + 1 < 255 ? newim[id].b + 1 : 255;
                }

                
                //ENERGY
                //newim[id + imdir] = (uchar4)(0, 0, 0, 255);
                //newim[id + imdir].g = evol[elw + 9] * 2 < 255 ? evol[dir + 9] * 10 : 255;
                //newim[id] = (uchar4)(0, 0, 0, 255);
                //newim[id].g = evol[elw + 9] * 2 < 255 ? evol[dir + 9] * 10 : 255;
            }
            else
            {
                evol[elw + 9] -= 5;
                newim[id] = lastim[id];

                //ENERGY
                //newim[id] = (uchar4)(0, 0, 0, 255);
                //newim[id].g = evol[elw + 9] * 2 < 255 ? evol[dir + 9] * 10 : 255;
            }
            //newim[id + imdir] = lastim[id];
            //newim[id] = lastim[id];
            
            
        }
        }
        //newim[id] = (uchar4)(0, 0, 0, 255);
        //lastim[id] = (uchar4)(0, 0, 0, 255);
    }
    else
        newim[id] = (uchar4)(0, 0, 0, 255);

}


void Mul(float * vec2, float * vec1, float * matrix, int width, int len, int index, int offset)
{
    for (unsigned int tx = 0; tx < len; tx++)
    {
        float value = 0;
        for (unsigned int k = 0; k < width; k++) {
            value += matrix[k * width + tx + offset] * vec1[index * 11 + k];
        }
        vec2[index * 11 + tx] = value;
    }
}

void CopyVec(float * vec1, float * vec2, int id)
{
    for(unsigned int i = 0; i < 11; i++)
        vec1[id * 11 + i] = vec2[id * 11 + i];
}

float rand(float x, float y, float z) {
    //float ptr = 0.0f;
    //return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
    float rnd = cos(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f);
    return (rnd - (int)rnd);
}