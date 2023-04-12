using Cloo;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace AI
{
    internal class GameOfLife
    {
        Game game;

        Color[] col;
        Texture2D oTex;
        Texture2D nTex;

        ComputeBuffer<Color> ot;
        ComputeBuffer<Color> nt;
        ComputeBuffer<float> n;

        ComputeContext Context;
        List<ComputeDevice> Devs = new List<ComputeDevice>();
        ComputeProgram prog = null;
        ComputeKernel kernelVecSum;
        ComputeCommandQueue Queue;
        Effect basicEffect;

        Color[] arrC;
        GCHandle arrCHandle;

        public GameOfLife(Game game)
        {
            this.game = game;

            ComputeContextPropertyList Properties = new ComputeContextPropertyList(ComputePlatform.Platforms[0]);
            Context = new ComputeContext(ComputeDeviceTypes.All, Properties, null, IntPtr.Zero);

            string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"Life.cl");
            StreamReader sr = new StreamReader(path);
            Devs.Add(ComputePlatform.Platforms[0].Devices[0]);
            try
            {
                prog = new ComputeProgram(Context, sr.ReadToEnd()); prog.Build(Devs, "", null, IntPtr.Zero);
            }
            catch (Exception ex)
            {
                string buildLog = prog.GetBuildLog(Devs[0]);
                string msg = ex.Message;
            }

            basicEffect = game.Content.Load<Effect>("Blur");

            int x = game.GraphicsDevice.Viewport.Width / 4;
            int y = game.GraphicsDevice.Viewport.Height / 4;
            oTex = new Texture2D(game.GraphicsDevice, x, y);
            nTex = new Texture2D(game.GraphicsDevice, x, y);

            Random rnd = new Random();
            col = new Color[x * y];
            for (int i = 0; i < col.Length; i++)
            {
                if (rnd.Next(2) == 1)
                {
                    col[i] = new Color(0, 150, 0, 255);
                }
                else
                    col[i] = Color.Black;
            }

            oTex.SetData(col);
            nTex.SetData(col);


            kernelVecSum = prog.CreateKernel("floatVectorSum");
            Queue = new ComputeCommandQueue(Context, Cloo.ComputePlatform.Platforms[0].Devices[0], Cloo.ComputeCommandQueueFlags.None);

            ot = new ComputeBuffer<Color>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, col);
            nt = new ComputeBuffer<Color>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, col);
            n = new ComputeBuffer<float>(Context,
                ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, new float[] { game.GraphicsDevice.Viewport.Width / 4 });

            arrC = new Color[col.Length];
            arrCHandle = GCHandle.Alloc(arrC, GCHandleType.Pinned);
        }

        public void Update()
        {
            ot = nt;

            kernelVecSum.SetMemoryArgument(0, ot);
            kernelVecSum.SetMemoryArgument(1, nt);
            kernelVecSum.SetMemoryArgument(2, n);

            Queue.Execute(kernelVecSum, null, new long[] { col.Length }, null, null);

            Queue.Read<Color>(nt, true, 0, col.Length, arrCHandle.AddrOfPinnedObject(), null);

            col = arrC;
            nTex.SetData(arrC);
        }
        public void Draw(SpriteBatch spriteBatch)
        {
            Vector2 uv = new Vector2(4f / game.GraphicsDevice.Viewport.Width, 4f / game.GraphicsDevice.Viewport.Height);
            Vector4 col = new Vector4(0, 200 / 255f, 0, 1);
            basicEffect.Parameters[0].SetValue(uv);
            basicEffect.Parameters[1].SetValue(col);

            spriteBatch.Begin(samplerState: SamplerState.PointClamp, effect: basicEffect);
            //spriteBatch.Begin(samplerState: SamplerState.PointClamp);
            //spriteBatch.Begin();

            spriteBatch.Draw(nTex, new Rectangle(0, 0, game.GraphicsDevice.Viewport.Width, game.GraphicsDevice.Viewport.Height), Color.White);

            spriteBatch.End();
        }
    }
}
