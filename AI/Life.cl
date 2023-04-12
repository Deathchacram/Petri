__kernel void
floatVectorSum(__global uchar4 * orig,
__global uchar4 * evol,
__global float * size)
{
    int i = get_global_id(0);
    int w = size[0];
    int x = orig[i-1].g + orig[i+1].g + orig[i-1-w].g + orig[i-w].g + orig[i+1-w].g + orig[i-1+w].g + orig[i+w].g + orig[i+1+w].g;
    x /= 255;
    if(i < w+1 || i > get_global_size(0) - w - 1)
        evol[i] = (uchar4)(0, 0, 0, 255);
    else
    {
        if(orig[i].g != 0)
        {
            if(x > 1)
            {
                if(x < 4)
                {
                    evol[i] = (uchar4)(255, 255, 255, 255);
                }
                else
                    evol[i] = (uchar4)(0, 0, 0, 255);
            }
            else
                evol[i] = (uchar4)(0, 0, 0, 255);
        }
        else if(x == 3)
            evol[i] = (uchar4)(255, 255, 255, 255);
        else
            evol[i] = (uchar4)(0, 0, 0, 255);
    }
}