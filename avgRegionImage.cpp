#include "mex.h" /* Always include this */
#include "math.h"

void mexFunction(int nlhs, mxArray *plhs[],             /* Output variables */
                 int nrhs, const mxArray *prhs[])       /* Input variables */
{
  


    //Declarations
    unsigned char *image;
    unsigned char *mask;
    double *outputImage;
    unsigned char pixel;
    int rows, cols;
    double sum = 0, avg = 0;
    
    int backgroundSize = 10;//(int)(mxGetScalar(mxGetField(prhs[2], 0, "backgroundSize")));
    int guardSize = 4;//(int)(mxGetScalar(mxGetField(prhs[2], 0, "guardSize")));
    int padSize = 0;//(int)(mxGetScalar(mxGetField(prhs[2], 0, "padSize")));
           
    // Get image as unsigned uint8 pointer as well as dimensions
    image = (unsigned char *)mxGetData(prhs[0]);
    mask = (unsigned char *)mxGetData(prhs[1]);
    
    // Get image/mask dimensions
    rows = mxGetN(prhs[0]);
    cols = mxGetM(prhs[0]);
    
    // Create an output image array
    plhs[0] = mxCreateNumericMatrix(cols, rows, mxDOUBLE_CLASS, false);
    outputImage = (double *) mxGetPr(plhs[0]);
    
    // Run through image and process it
    for (int i = 0 + padSize; i < rows - padSize; i++)
    {
        for (int j = 0 + padSize; j < cols - padSize; j++)
        {
           pixel = image[j + cols*i];

           if(mask[j + cols*i] > 0)
            {
                for(int x = -floor(backgroundSize/2); x <= floor(backgroundSize/2); x++)
                {
                    for(int y = -floor(backgroundSize/2); y <= floor(backgroundSize/2); y++)
                    {
                        sum += (int) image[(j+x) + cols*(i+y)];
                    }
                }

                for(int x = -floor(guardSize/2); x <= floor(guardSize/2); x++)
                {
                    for(int y = -floor(guardSize/2); y <= floor(guardSize/2); y++)
                    {
                        sum -= (int) image[(j+x) + cols*(i+y)];
                    }
                }
                
                outputImage[j + cols*i] = sum/(backgroundSize*backgroundSize - guardSize*guardSize);
                
                sum = 0;
            }
            else
             outputImage[j + cols*i] = 0;
        } 
    }
    
    return;
}