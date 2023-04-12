using Cloo;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Myra.Graphics2D.UI;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Xml.Serialization;

namespace AI
{
    public class BaseEvolution
    {
        Game1 game;
        public Desktop desktop;

        Texture2D oTex; //old screen
        Texture2D nTex; //new screen
        Effect basicEffect;

        public ComputeBuffer<Color> lastIm;
        public ComputeBuffer<Color> newIm;
        public ComputeBuffer<float> vector1;   //first vector
        public ComputeBuffer<float> vector2;   //second vector
        public ComputeBuffer<float> matrix;    //evol
        public ComputeBuffer<float> size;

        ComputeContext Context;
        List<ComputeDevice> Devs = new List<ComputeDevice>();
        ComputeProgram prog = null;
        ComputeKernel kernelVecSum;
        ComputeCommandQueue Queue;

        int x = 0, y = 0;
        public Color[] col;    //screen
        Color[] col2;    //screen
        float[] vec1;   //vector2
        float[] vec2;   //vector2
        public float[] mx;     //matrix
        Color[] result;
        //float[] result;
        float[] sz;     //size
        GCHandle resultCHandle;
        public BaseEvolution() { }
        public BaseEvolution(Game1 game)
        {
            this.game = game;

            basicEffect = game.Content.Load<Effect>("Blur");

            ComputeContextPropertyList Properties = new ComputeContextPropertyList(ComputePlatform.Platforms[0]);
            Context = new ComputeContext(ComputeDeviceTypes.All, Properties, null, IntPtr.Zero);

            //string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"UnicelluarEvol.cl");
            //string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"Mushroom.cl");
            string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"BaseEvol4_v3.cl");
            StreamReader sr = new StreamReader(path);

            Devs.Add(ComputePlatform.Platforms[0].Devices[0]);
            try
            {
                prog = new ComputeProgram(Context, sr.ReadToEnd()); prog.Build(Devs, "", null, IntPtr.Zero);
            }
            catch (Exception ex)
            {
                //string buildLog = prog.GetBuildLog(Devs[0]);
                string msg = ex.Message;
            }

            string buildLog = prog.GetBuildLog(Devs[0]);
            kernelVecSum = prog.CreateKernel("floatVectorSum");

            Setup();
        }
        private void Setup()
        {
            game.pause = true;

            x = game.GraphicsDevice.Viewport.Width / 4;
            y = game.GraphicsDevice.Viewport.Height / 4;

            oTex = new Texture2D(game.GraphicsDevice, x, y);
            nTex = new Texture2D(game.GraphicsDevice, x, y);

            col = new Color[x * y];
            col2 = new Color[x * y];
            //sz = new float[] { x, 7, 1, 427, x * y };   //screen, vec in, vision, evol wi, vec out
            sz = new float[] { x, 7, 1, 203, x * y };   //screen, vec in, vision, evol wi, vec out
            //sz = new float[] { x, 5, 1, 132, x * y };      //screen, vec in, vision, evol wi, vec out
            //sz = new float[] { x, 8, 1, 119, x * y };   //screen, vec in, vision, evol wi, vec out
            vec1 = vec2 = new float[(int)sz[1] * x * y];
            mx = new float[(int)sz[3] * x * y];

            CreateWorld();

            Queue = new ComputeCommandQueue(Context, ComputePlatform.Platforms[0].Devices[0], ComputeCommandQueueFlags.None);

            lastIm = new ComputeBuffer<Color>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, col);
            newIm = new ComputeBuffer<Color>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, col2);
            vector1 = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, vec1);
            vector2 = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, vec2);
            matrix = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, mx);
            size = new ComputeBuffer<float>(Context,
                ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, sz);

            result = new Color[col.Length];
            //resultM = new float[mx.Length];
            resultCHandle = GCHandle.Alloc(result, GCHandleType.Pinned);

            //game.pause = false;
        }
        public void Update()
        {
            kernelVecSum.SetMemoryArgument(0, lastIm);
            kernelVecSum.SetMemoryArgument(1, newIm);
            kernelVecSum.SetMemoryArgument(2, vector1);
            kernelVecSum.SetMemoryArgument(3, vector2);
            kernelVecSum.SetMemoryArgument(4, matrix);
            kernelVecSum.SetMemoryArgument(5, size);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            Queue.Execute(kernelVecSum, null, new long[] { col.Length }, null, null);

            Queue.Read<Color>(newIm, true, 0, col.Length, resultCHandle.AddrOfPinnedObject(), null);
            //Queue.Read<float>(vector2, true, 0, col.Length, resultCHandle.AddrOfPinnedObject(), null);

            sw.Stop();
            lastIm = newIm;

            col = result;
            nTex.SetData(result);
        }
        public void Draw(SpriteBatch spriteBatch)
        {
            Vector2 uv = new Vector2(1f / x, 1f / y);
            //Vector4 col = new Vector4(0, 200 / 255f, 0, 1);
            basicEffect.Parameters[0].SetValue(uv);
            //basicEffect.Parameters[1].SetValue(col);


            spriteBatch.Begin(samplerState: SamplerState.PointClamp, effect: basicEffect);
            //spriteBatch.Begin(samplerState: SamplerState.PointClamp);
            //spriteBatch.Begin();

            spriteBatch.Draw(nTex, new Rectangle(0, 0, game.GraphicsDevice.Viewport.Width, game.GraphicsDevice.Viewport.Height), Color.White);

            spriteBatch.End();
        }

        private void CreateWorld()
        {
            Random rnd = new Random();

            /*for (int l = 0; l < col.Length; l++)
            {
                bool b = rnd.Next(2) == 1 ? true : false;
                if (b)
                {
                    float[] f = new float[(int)sz[3]];
                    for (int i = 0; i < (int)sz[3]; i++)
                    {
                        mx[(int)sz[3] * l + i] = (float)(rnd.NextDouble() * 2f - 1f);
                        f[i] = mx[(int)sz[3] * l + i];
                    }
                    //col[index] = col2[index] = Color.LimeGreen;
                    col[l] = col2[l] = new Color(127, 127, 0, 255);

                    mx[(int)sz[3] * l + 4] = rnd.Next(16, 25); //TTL
                    mx[(int)sz[3] * l + 5] = rnd.Next(15, 25); //Energy
                    mx[(int)sz[3] * l + 6] = rnd.Next(40, 70);

                }
                else
                    col[l] = col2[l] = Color.Black;
            }*/
            for (int i = 0; i < vec1.Length; i++)
                vec1[i] = vec2[i] = 0;

            for (int i = 0; i < col.Length; i++)
                col[i] = col2[i] = Color.Black;
            for (int m = 0; m < 2000; m++)
            {
                int index = x * y / 2 + rnd.Next(-x * y / 2, x * y / 2);
                //int index = 0;

                float[] f = new float[(int)sz[3]];
                for (int i = 0; i < (int)sz[3]; i++)
                {
                    mx[(int)sz[3] * index + i] = (float)(rnd.NextDouble() * 2f - 1f);
                    f[i] = mx[(int)sz[3] * index + i];
                }
                col[index] = col2[index] = new Color(127, 127, 0, 255);

                mx[(int)sz[3] * index + 4] = 100;
                mx[(int)sz[3] * index + 5] = 100;
                //mx[(int)sz[3] * index + 6] = rnd.Next(40, 70);
            }

            oTex.SetData(col);
            nTex.SetData(col);
        }

        void Save()
        {
            game.pause = true;

            game.onClick += () =>
            {
                TextBox tb = (TextBox)desktop.Widgets[2];
                string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), $@"Content/{tb.Text}.dat");

                Point pos = Mouse.GetState().Position;
                int w = game.GraphicsDevice.Viewport.Width / x; //get pixel
                int i = (pos.Y - 4) / w * x  + pos.X / w;
                i *= (int)sz[3];

                float[] fx = new float[(int)sz[3]];
                GCHandle fxHandle = GCHandle.Alloc(fx, GCHandleType.Pinned);

                Queue.Read<float>(matrix, true, i, (int)sz[3], fxHandle.AddrOfPinnedObject(), null);

                /*float[] res = new float[(int)sz[3]];
                for (int l = i; l < fx.Length; l++)
                    res[l - i] = fx[l];*/

                fx[4] = 100;
                fx[5] = 100;

                var bf = new XmlSerializer(typeof(float[]));
                File.Delete(path);
                FileStream file = File.Open(path, FileMode.OpenOrCreate);
                bf.Serialize(file, fx);
                file.Close();

                fxHandle.Free();

                game.onClick = null;
            };

        }
        void Load()
        {
            game.pause = true;

            game.onClick += () =>
            {
                TextBox tb = (TextBox)desktop.Widgets[2];

                string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), $@"Content/{tb.Text}.dat");

                Point pos = Mouse.GetState().Position;
                int w = game.GraphicsDevice.Viewport.Width / x;
                if (pos.X > 0 && pos.Y > 0 && pos.X / w < x && pos.Y / 2 < y)
                { //get pixel
                    int i = (pos.Y - 4) / w * x + pos.X / w;
                    i *= (int)sz[3];

                    float[] fx = new float[(int)sz[3]];

                    var bf = new XmlSerializer(typeof(float[]));
                    FileStream file = File.Open(path, FileMode.OpenOrCreate);
                    fx = bf.Deserialize(file) as float[];
                    file.Close();

                    GCHandle fxHandle = GCHandle.Alloc(mx, GCHandleType.Pinned);

                    Queue.Read<float>(matrix, true, 0, mx.LongLength, fxHandle.AddrOfPinnedObject(), null);

                    float[] res = new float[(int)sz[3]];
                    for (int l = i; l < i + (int)sz[3]; l++)
                        mx[l] = fx[l - i];

                    matrix = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, mx);
                    fxHandle.Free();
                }
                game.onClick = null;
            };
        }
        public void Menu()
        {
            desktop = new Desktop();

            int x = game.GraphicsDevice.Viewport.Width;
            int y = game.GraphicsDevice.Viewport.Height;

            TextButton button1 = new TextButton
            {
                Left = (int)(x - 150),
                Top = (int)(20),
                Width = (int)(120f),
                Height = (int)(20f),
                Text = "Save"
            };
            button1.TouchUp += (s, a) => Save();
            desktop.Widgets.Add(button1);
            TextButton button2 = new TextButton
            {
                Left = (int)(x - 150),
                Top = (int)(50),
                Width = (int)(120f),
                Height = (int)(20f),
                Text = "load"
            };
            desktop.Widgets.Add(button2);
            button2.TouchUp += (s, a) => Load();
            var tb = new TextBox
            {
                Left = x - 150,
                Top = 80,
                Width = 100,
                Text = "net1"
            };
            desktop.Widgets.Add(tb);
            TextButton button3 = new TextButton
            {
                Left = (int)(x - 150),
                Top = (int)(110),
                Width = (int)(120f),
                Height = (int)(20f),
                Text = "Pause"
            };
            desktop.Widgets.Add(button3);
            button3.TouchDown += (s, a) => { game.pause = !game.pause; };
            TextButton button4 = new TextButton
            {
                Left = (int)(x - 150),
                Top = (int)(140),
                Width = (int)(120f),
                Height = (int)(20f),
                Text = "Reset"
            };
            desktop.Widgets.Add(button4);
            button4.TouchDown += (s, a) => { Setup(); };
            TextButton button5 = new TextButton
            {
                Left = (int)(x - 150),
                Top = (int)(170),
                Width = (int)(120f),
                Height = (int)(20f),
                Text = "Vision"
            };
            desktop.Widgets.Add(button5);
            button5.TouchDown += (s, a) =>
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();

                /*if (sz[2] == 0)
                    sz[2] = 1;
                else
                    sz[2] = 0;*/
                sz[2] = (sz[2] + 1) % 3;

                for (int i = 0; i < col.Length; i++)
                {
                    if (col[i] != Color.Black)
                        col[i] = col2[i] = new Color(127, 127, 0, 255);
                }
                //lastIm = new ComputeBuffer<Color>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, col);
                //newIm = new ComputeBuffer<Color>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, col2);

                size = new ComputeBuffer<float>(Context,
                    ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, sz);

            };
        }
    }
}
