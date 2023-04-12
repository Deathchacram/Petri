using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Myra;
using Myra.Graphics2D.UI;
using System;

namespace AI
{
    public class Game1 : Game
    {
        Desktop desktop;
        private GraphicsDeviceManager _graphics;
        private SpriteBatch spriteBatch;

        public bool pause = true;
        public bool draw = true;
        public bool click = true;
        public delegate void OnClick();
        public OnClick onClick;

        GameOfLife gf;
        BaseEvolution be;

        int counter = 0;

        public Game1()
        {
            _graphics = new GraphicsDeviceManager(this);

            MyraEnvironment.Game = this;

            _graphics.IsFullScreen = false;
            _graphics.PreferredBackBufferWidth = 800;
            _graphics.PreferredBackBufferHeight = 480;
            _graphics.ApplyChanges();

            Content.RootDirectory = "Content";
            IsMouseVisible = true;
        }

        protected override void Initialize()
        {
            base.Initialize();
        }

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
            //gf = new GameOfLife(this);
            be = new BaseEvolution(this);
            be.Menu();

            /*ComputeContextPropertyList Properties = new ComputeContextPropertyList(ComputePlatform.Platforms[0]);
            ComputeContext Context = new ComputeContext(ComputeDeviceTypes.All, Properties, null, IntPtr.Zero);

            string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"Mul.cl");
            StreamReader sr = new StreamReader(path);

            ComputeProgram prog = null;
            ComputeKernel kernelVecSum;
            ComputeCommandQueue Queue;

            List<ComputeDevice> Devs = new List<ComputeDevice>();
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

            int x = GraphicsDevice.Viewport.Width / 4;
            int y = GraphicsDevice.Viewport.Height / 4;

            Color[] col;    //screen
            Color[] col2;    //screen
            float[] vec1;   //vector2
            float[] vec2;   //vector2
            float[] mx;     //matrix
            Color[] result;
            //float[] result;
            float[] sz;     //size
            GCHandle resultCHandle;

            sz = new float[] { x, 3, 0, 23 };   //screen, vec in, vision, evol wi, vec out
            vec1 = vec2 = new float[] { 1, 2, 0 };
            mx = new float[] { 0, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6 };

            Queue = new ComputeCommandQueue(Context, ComputePlatform.Platforms[0].Devices[0], ComputeCommandQueueFlags.None);

            ComputeBuffer<float> vector1 = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, vec1);
            ComputeBuffer<float> vector2 = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, vec2);
            ComputeBuffer<float> matrix = new ComputeBuffer<float>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, mx);
            ComputeBuffer<float> size = new ComputeBuffer<float>(Context,
                ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, sz);

            float[] resultM = new float[3];
            resultCHandle = GCHandle.Alloc(resultM, GCHandleType.Pinned);

            kernelVecSum.SetMemoryArgument(0, vector1);
            kernelVecSum.SetMemoryArgument(1, vector2);
            kernelVecSum.SetMemoryArgument(2, matrix);
            kernelVecSum.SetMemoryArgument(3, size);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            Queue.Execute(kernelVecSum, null, new long[] { 1 }, null, null);

            Queue.Read<float>(vector2, true, 0, 3, resultCHandle.AddrOfPinnedObject(), null);*/
            //Queue.Read<float>(vector2, true, 0, col.Length, resultCHandle.AddrOfPinnedObject(), null);
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

            if(!click && Mouse.GetState().LeftButton == ButtonState.Pressed && onClick != null)
            {
                onClick();
                click = true;
            }
            else if(click && Mouse.GetState().LeftButton == ButtonState.Released)
                click = false;


            counter++;
            if (counter % 5 == 0 && !pause)
            {
                //gf.Update();
                be.Update();
            }
            if (counter % 20 == 0 && !pause)
            {
                //GC.Collect();
                //GC.WaitForPendingFinalizers();
            }
            counter %= 60;
            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {

            //GraphicsDevice.Clear(Color.CornflowerBlue);
            if (draw)
                be.Draw(spriteBatch);
            be.desktop.Render();

            base.Draw(gameTime);
        }
    }
}