// using System.Reflection.PortableExecutable;
// using System.Numerics;
// using System.Reflection.Metadata;  //assembly报错
using System.Linq.Expressions;
// using System.Reflection.PortableExecutable; //assembly报错
using System.Runtime.CompilerServices;
// using System.Diagnostics;
using System;
using System.Collections.Generic;
using System.IO; //必须要引入该命名空间，因为需要使用它的File类
using UnityEditor;
using UnityEngine;

//socket
using System.Net.Sockets;
using System.Text;

namespace DeepLearning
{
    public class MyModel: NativeNetwork
    {
        public string file = "";
        public List<Matrix> M { get; private set; } = new List<Matrix>(); // 存放所有读取的数据
        public List<Matrix> M1 { get; private set; } = new List<Matrix>();
        public List<Matrix> M2 { get; private set; } = new List<Matrix>();
        private int lineDim = 0; // 每帧的维度
        private int framesNum = 0; //总帧数长度
        // public float framerate = 30f;

        //socket
        string toPython = ""; // 发送socket的string
        
        // public int FeedFeaturesNum = 0;
        // public int ReadFeaturesNum = 0;
        // public int ReadFeaturesNum2 = 0;
        NetworkStream stream;
        string server = "127.0.0.1";
        int port = 5000;
        

        
        
        private void start() {
            
        }

        #if UNITY_EDITOR
        public override void InspectorDerived(Editor editor) {
            if(Utility.GUIButton("Restart History", Color.grey, Color.white)) {
                if(enableSocket) {SendData(sendRestart:true);}
                RestartHistory();
            }
            editor.DrawDefaultInspector();
            // if(Utility.GUIButton("Import Parameters", Color.grey, Color.white)) {
            //     Parameters = Parameters.Import(Folder);
            // }
            // if(Utility.GUIButton("Export Parameters", Color.grey, Color.white)) {
            //     Parameters.Export(Folder);
            // }
        }
        #endif
        
        private void ReadLocalData() {
            string file_path = Folder + file;
            foreach (string s in File.ReadLines(file_path)) {
                string[] outputs = s.Split(' ');
                lineDim = outputs.Length;
                
                Matrix m = new Matrix(lineDim, 1);
                for (int idx = 0; idx < lineDim; idx++) {
                    m.SetValue(idx, 0, float.Parse(outputs[idx]));
                }
                M.Add(m);
                framesNum += 1;
                if(stopReadFrame>0 && framesNum>stopReadFrame) break;
            }
            print("Local data features: "+lineDim);
        }

        protected override void LoadDerived() {
            countFrame = 0;
            X = CreateMatrix(9999, 1, "X");
            ReadLocalData();

            if(enableSocket){
                TcpClient client = new TcpClient(server, port);
                stream = client.GetStream();
            }
            
        }
        protected override void UnloadDerived() {

        }
        protected override void PredictDerived() {
            int currentIndex = countFrame + startIndex;

            if(!enableSocket) {
                if (currentIndex >= framesNum) { 
                    finishReading = true;
                    UnityEngine.Debug.Log("读取结束");
                }
                else {
                    Y = M[currentIndex];
                }
                
            }
            if(enableSocket) {
                SendData();
                Y = ReceiveData();
            }
            // countFrame += 1;
        }

        private void SendData(bool sendRestart=false) {
            byte[] data;
            if(sendRestart) {
                data = Encoding.ASCII.GetBytes("-1"); toPython=""; 
                stream.Write(data, 0, data.Length);
            }
            else {
                for(int i = 0; i<=GetPivot(); i++) {
                    toPython += String.Format("{0:F5}", X.GetValue(i,0)) + " ";
                }
                // Debug.Log(GetPivot()+1);
                data = Encoding.ASCII.GetBytes(toPython); toPython=""; 
                stream.Write(data, 0, data.Length);
            }
        }

        private Matrix ReceiveData(){
            byte[] receivedBytes = new byte[500000];
            int bytes = stream.Read(receivedBytes, 0, receivedBytes.Length);
            string receivedString = Encoding.ASCII.GetString(receivedBytes, 0, bytes);
            string[] receivedStr = receivedString.Split(',');
            // print("接收字节："+bytes.ToString()+"，接收字符串："+str.Length);

            List<float> receivedFloats = new List<float>();
            int i=0;
            while(true) {
                // Debug.Log(receivedStr[i]);
                if(receivedStr[i]=="RESTART") {RestartHistory();}
                else if(receivedStr[i]=="END") {break;}
                else {
                    receivedFloats.Add(float.Parse(receivedStr[i]));
                }
                i+=1;
            }
            return Matrix.FromList(receivedFloats);
        }

        private void RestartHistory() {
            countFrame = 0;
        }
    }
}