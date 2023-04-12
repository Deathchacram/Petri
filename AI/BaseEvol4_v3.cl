#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable


void Mul(float *, float *, float *, int, int, int, int);
void CopyVec(float *, float *, int, int);
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
    
    if(evol[elw + 5] <= 0) //if no energy => die
        evol[elw + 4] = 0;
    if(evol[elw + 4] <= 0)                      //if die => destroy body
        evol[elw + 5] = 0;

    if(id < scrw*2 || id > size[4] - scrw*2)
    {
        evol[elw + 4] = 0;
        evol[elw + 5] = 0;
    }

    if(evol[elw + 4] > 1)
    {
        if(size[2] == 0)
        {
        float red = 0;
        float green = 0;
        float blue = 0;
        
        for(int i = 7; i < (int)size[3] / 3; i+=3)
            red += evol[elw + i];
        for(int i = (int)size[3] / 3 + 1; i < (int)size[3] / 3 * 2; i+=3)
            green += evol[elw + i];
        for(int i = (int)size[3] / 3 * 2 + 1; i < (int)size[3]; i+=3)
            blue += evol[elw + i];

        red = abs((int)red) * 50;
        green = abs((int)green) * 50;
        blue = abs((int)blue) * 50;

        color = (uchar4)(red, green, blue, 255);
        }
        

        vec1[vecw + 0] = 0;        //8 neighbours
        vec1[vecw + 1] = 0;
        vec1[vecw + 2] = 0;
        vec1[vecw + 3] = 0;

        int f1 = elw - scrw * (int)size[3];
        int f2 = elw + (int)size[3];
        int f3 = elw + scrw * (int)size[3];
        int f4 = elw - (int)size[3];
        
        int sf1 = evol[f1 + 4] > 0 ? 1 : 0;
        int sf2 = evol[f2 + 4] > 0 ? 1 : 0;
        int sf3 = evol[f3 + 4] > 0 ? 1 : 0;
        int sf4 = evol[f4 + 4] > 0 ? 1 : 0;

        for(int i = 7; i < size[3]; i+=2)
        {
            vec1[vecw    ] += (evol[f1 + i] - evol[elw + i]) * sf1;        //4 neighbours
            vec1[vecw + 1] += (evol[f2 + i] - evol[elw + i]) * sf2;
            vec1[vecw + 2] += (evol[f3 + i] - evol[elw + i]) * sf3;
            vec1[vecw + 3] += (evol[f4 + i] - evol[elw + i]) * sf4;
        }
        
        
        vec1[vecw    ] = abs((int)(vec1[vecw  ]   * 5));  //genocode
        vec1[vecw + 1] = abs((int)(vec1[vecw+1]   * 5));  //genocode
        vec1[vecw + 2] = abs((int)(vec1[vecw+2]   * 5));  //genocode
        vec1[vecw + 3] = abs((int)(vec1[vecw+3]   * 5));  //genocode

        vec1[vecw + 4] = evol[elw + 5];         //energy
        vec1[vecw + 5] = evol[elw + 4] - 1;   //life time
        //vec1[vecw + 5] = 0;   //life time
        vec1[vecw + 6] = 1;


        evol[elw + 0] = 0;
        evol[elw + 4] = evol[elw + 4] - 1;
        evol[elw + 5] = evol[elw + 5] - 1;
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        Mul(vec2, vec1, evol, 7, 15, id, 7 + elw);
        CopyVec(vec1, vec2, vecw, size[1]);
        Mul(vec2, vec1, evol, 15, 7, id, 112 + 0 + elw);

        /*Mul(vec2, vec1, evol, 7, 7, id, 7 + elw);
        CopyVec(vec1, vec2, vecw, size[1]);
        Mul(vec2, vec1, evol, 7, 7, id, 56 + elw);
        CopyVec(vec1, vec2, vecw, size[1]);
        Mul(vec2, vec1, evol, 7, 7, id, 105 + elw);
        CopyVec(vec1, vec2, vecw, size[1]);
        Mul(vec2, vec1, evol, 7, 7, id, 154 + elw);*/

        //CopyVec(vec1, vec2, vecw);
        //Mul(vec2, vec1, evol, 7, 7, id, 154 + elw);
        //CopyVec(vec1, vec2, vecw);
        //Mul(vec2, vec1, evol, 7, 7, id, 203 + elw);


        barrier(CLK_GLOBAL_MEM_FENCE);

        //DIRECTION

        float4 vv = (float4)(0, 0, 0, 0);
        vv.r = vec2[vecw + 0];
        vv.g = vec2[vecw + 1];
        vv.b = vec2[vecw + 2];
        vv.a = vec2[vecw + 3];

        int dir = elw - scrw * size[3];  

        vv =  normalize(vv);
        float max = vv.r;
        if(max < vv.g) {max = vv.g; dir = elw + size[3]; }
        if(max < vv.b) {max = vv.b; dir = elw + scrw * size[3]; }
        if(max < vv.a) {max = vv.a; dir = elw - size[3]; }

        int x = dir;
        int imdir = (dir - elw) / size[3];
        
        //ACTION

        int action = 0;

        //action = fmod((float)abs((int)(vec2[vecw + 4])), (float)3);

        vv.r = vec2[vecw + 4];
        vv.g = vec2[vecw + 5];
        vv.b = vec2[vecw + 6];
        //vv.a = vec2[vecw + 3];
        vv.a = 0;

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

        if(action == 0)                    //PHOTOSYNTHESIS
        {
            evol[elw + 0] = 1;
            evol[elw + 5] += 4;
            evol[elw + 6] = (evol[elw + 6] - 1) > -127 ? evol[elw + 6] - 1 : -127;
            
            newim[id] = lastim[id];
            //newim[id] = (uchar4)(127, 127, 0, 255);
            
            //GENOCODE
            if(size[2] == 0)
                newim[id] = color;

            //GREEN OR RED
            if(size[2] == 1)
            {
                int gre = 150 + evol[elw + 6] * 1 > 30 ? 150 + evol[elw + 6] * 1 : 30;
                gre = gre < 220 ? gre : 220;

                int re = 150 - evol[elw + 6] * 1 > 30 ? 150 - evol[elw + 6] * 1 : 30;
                re = re < 220 ? re : 220;

                newim[id] = (uchar4)(gre, re, 0, 255);
            }

            //ENERGY
            if(size[2] == 2)
            {
                int gre = evol[elw + 5] * 4 + 50 > 0 ? evol[elw + 5] * 4 + 50 : 0;
                gre = gre < 255 ? gre : 255;

                newim[id] = (uchar4)(0, gre, 0, 255);
            }
            
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 1)                             //ATTACK
        {
            if(evol[dir + 4] != 0)
                evol[dir + 6] = (evol[elw + 6] + 25) < 127 ? evol[elw + 6] + 25 : 127;
            evol[elw + 6] = 0;

            evol[dir + 0] = 1;
            evol[elw + 0] = 1;

            evol[dir + 1] = evol[elw + 1];
            evol[elw + 1] = 0;

            evol[dir + 4] = evol[elw + 4];
            evol[elw + 4] = 0;

            evol[dir + 5] = evol[elw + 5] + evol[dir + 5]/2 - 0;
            evol[elw + 5] = 0;

            for(int i = 7; i < size[3]; i++)
            {
                evol[dir + i] = evol[elw + i];
            }
            newim[id] = (uchar4)(0, 0, 0, 255);
            newim[id + imdir] = lastim[id];
            //newim[id + imdir] = (uchar4)(127, 127, 0, 255);
            
            //GENOCODE
            if(size[2] == 0)
                newim[id + imdir] = color;

            //GREEN OR RED
            if(size[2] == 1)
            {
                int gre = 127 + evol[dir + 6] * 1 > 30 ? 127 + evol[dir + 6] * 1 : 30;
                gre = gre < 220 ? gre : 220;

                int re = 127 - evol[dir + 6] * 1 > 30 ? 127 - evol[dir + 6] * 1 : 30;
                re = re < 220 ? re : 220;

                newim[id + imdir] = (uchar4)(gre, re, 0, 255);
            }

            //ENERGY
            if(size[2] == 2)
            {
                int gre = evol[dir + 5] * 4 + 50 > 0 ? evol[dir + 5] * 4 + 50 : 0;
                gre = gre < 255 ? gre : 255;

                newim[id + imdir] = (uchar4)(0, gre, 0, 255);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(action == 2)                    //BORN
        {
            if(evol[dir + 4] == 0)
            {
                evol[elw + 0] = 1;
                evol[dir + 0] = 1;

                evol[dir + 1] =  evol[elw + 1];

                float rnd = rand(evol[elw + 4], evol[elw + 5], id);
                evol[dir + 5] = evol[elw + 5] / 3;
                evol[elw + 5] = evol[elw + 5] / 3;

                evol[dir + 4] = 100;
                evol[dir + 6] = evol[elw + 6];

                for(int i = 7; i < size[3]; i++)
                {
                    evol[dir + i] = evol[elw + i];
                }
                //newim[id] = (uchar4)(127, 127, 0, 255);
                //newim[id + imdir] = (uchar4)(127, 127, 0, 255);

                //GENOCODE
                if(size[2] == 0)
                {
                    newim[id + imdir] = color;
                    newim[id] = color;
                }

                //GREEN OR RED
                if(size[2] == 1)
                {
                    int gre = 150 + evol[elw + 6] * 2 > 30 ? 150 + evol[elw + 6] * 2 : 30;
                    gre = gre < 220 ? gre : 220;

                    int re = 150 - evol[elw + 6] * 2 > 30 ? 150 - evol[elw + 6] * 2 : 30;
                    re = re < 220 ? re : 220;

                    newim[id + imdir] = (uchar4)(gre, re, 0, 255);
                    newim[id] = (uchar4)(gre, re, 0, 255);
                }
                
                //ENERGY
                if(size[2] == 2)
                {
                    int gre = evol[dir + 5] * 4 + 50 > 0 ? evol[dir + 5] * 4 + 50 : 0;
                    gre = gre < 255 ? gre : 255;

                    newim[id + imdir] = (uchar4)(0, gre, 0, 255);
                    
                    gre = evol[elw + 5] * 4 + 50 > 0 ? evol[elw + 5] * 4 + 50 : 0;
                    gre = gre < 255 ? gre : 255;

                    newim[id] = (uchar4)(0, gre, 0, 255);
                }

                if(abs((int)rnd) < 0.2)
                {
                    for(int i = 0; i < 5; i++)
                    {
                        rnd = rand(evol[elw + 4], evol[elw + 5], i);      //MUTATION
                        int s = abs((int)(rnd* (size[3] - 7) ));
                        evol[dir + 7 + s] += rand(i, s, id) / 100;
                    }
                }
            }
            else
            {
                evol[elw + 0] = 1;

                newim[id] = lastim[id];
                //newim[id] = (uchar4)(127, 127, 0, 255);
            }
            
            
        }
    }
    else
        newim[id] = (uchar4)(0, 0, 0, 255);

}

void Mul(float * vec2, float * vec1, float * matrix, int width, int len, int index, int offset)
{
    int id = index * len;
    for (int tx = 0; tx < width; tx++)
    {
        float value = 0;
        for (int k = 0; k <  len - 1; ++k) {
            value += matrix[tx * width + k + offset] * vec1[id + k];
        }
        float bias = matrix[tx * width + len - 1 + offset];
        vec2[id + tx] = 1 / (1 + exp(-(value + bias)));
    }
}

void CopyVec(float * vec1, float * vec2, int id, int len)
{
    for(unsigned int i = 0; i < len; i++)
        vec1[id + i] = vec2[id + i];
}

float rand(float x, float y, float z) {
    float rnd = cos(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f);
    return (rnd);
}