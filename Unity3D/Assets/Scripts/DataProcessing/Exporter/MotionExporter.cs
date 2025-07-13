#if UNITY_EDITOR
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEditor;
using System;
using System.Threading;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.ComponentModel;

public class MotionExporter : EditorWindow {

	public enum PIPELINE {Basketball, Quadruped};

	[Serializable]
	public class Asset {
		public string GUID = string.Empty;
		public bool Selected = true;
		public bool Exported = false;
	}

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public PIPELINE Pipeline = PIPELINE.Basketball; 

	public int FrameShifts = 0;
	public int FrameBuffer = 10;
	public bool WriteMirror = false;

	private string Filter = string.Empty;
	private Asset[] Assets = new Asset[0];
	[NonSerialized] private Asset[] Instances = null;

	private static bool Aborting = false;
	private static bool Exporting = false;

	private int Page = 0;
	private int Items = 25;

	private float Progress = 0f;
	private float Performance = 0f;

	private static string Separator = " ";
	private static string Accuracy = "F5";

	private MotionEditor Editor = null;
	private MotionEditor Editor2 = null;
	private int OnlyFirstXFrames = 0;
	private bool correct_twist = false;
	private string OutputFileName = "Output";
	private string ExportPath = "D:/MyData/Boxing/Motion Matching";
	private bool SalsaDance = false;
	private bool InterHuman = false;
	private int repeat = 1;

	[MenuItem ("AI4Animation/Exporter/Motion Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionExporter));
		Scroll = Vector3.zero;
	}
	
	public void OnInspectorUpdate() {
		Repaint();
	}
	
	public void Refresh() {
		if(Editor == null) {
			// Editor = GameObject.FindObjectOfType<MotionEditor>(); // ORIGINAL
			Editor = GameObject.Find("MotionEditor").GetComponent<MotionEditor>();
		}
		if(Editor2 == null) { //todo
			Editor2 = GameObject.Find("MotionEditor2").GetComponent<MotionEditor>();
			Debug.Log("成功加载MotionEditor2");
		}
		if(Editor != null && Assets.Length != Editor.Assets.Length) {
			Assets = new Asset[Editor.Assets.Length];
			for(int i=0; i<Editor.Assets.Length; i++) {
				Assets[i] = new Asset();
				Assets[i].GUID = Editor.Assets[i];
				Assets[i].Selected = true;
				Assets[i].Exported = false;
			}
			Aborting = false;
			Exporting = false;
			ApplyFilter(string.Empty);
		}
		if(Instances == null) {
			ApplyFilter(string.Empty);
		}
	}

	public void ApplyFilter(string filter) {
		Filter = filter;
		if(Filter == string.Empty) {
			Instances = Assets;
		} else {
			List<Asset> instances = new List<Asset>();
			for(int i=0; i<Assets.Length; i++) {
				if(Utility.GetAssetName(Assets[i].GUID).ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Assets[i]);
				}
			}
			Instances = instances.ToArray();
		}
		LoadPage(1);
	}

	public void LoadPage(int page) {
		Page = Mathf.Clamp(page, 1, GetPages());
	}

	public int GetPages() {
		return Mathf.CeilToInt(Instances.Length/Items)+1;
	}

	public int GetStart() {
		return (Page-1)*Items;
	}

	public int GetEnd() {
		return Mathf.Min(Page*Items, Instances.Length);
	}

	private string GetExportPath() {
		// string path = "D:/OneDrive - Durham University/UnityProjects/Sebastian Starke/AI4Animation/MyData";
		// return path;

		return ExportPath;
	}

	void OnGUI() {
		Refresh();

		if(Editor == null) {
			EditorGUILayout.LabelField("No editor available in scene.");
			return;
		}

		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Motion Exporter");
				}

				Utility.SetGUIColor(UltiDraw.White);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.FloatField("Export Framerate", Editor.TargetFramerate);
					EditorGUI.EndDisabledGroup();
					ExportPath = EditorGUILayout.TextField("Export Path", ExportPath);
					OutputFileName = EditorGUILayout.TextField("Output File Name", OutputFileName);
				}

				Pipeline = (PIPELINE)EditorGUILayout.EnumPopup("Pipeline", Pipeline);
				FrameShifts = EditorGUILayout.IntField("Frame Shifts", FrameShifts);
				FrameBuffer = Mathf.Max(1, EditorGUILayout.IntField("Frame Buffer", FrameBuffer));
				WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);
				OnlyFirstXFrames = EditorGUILayout.IntField("只导出前 X 帧", OnlyFirstXFrames);
				correct_twist = EditorGUILayout.Toggle("Correct twist", correct_twist);
				SalsaDance = EditorGUILayout.Toggle("Salsa dataset", SalsaDance);
				InterHuman = EditorGUILayout.Toggle("InterHuman dataset", InterHuman);
				repeat = EditorGUILayout.IntField("repeat", repeat);

				if(!Exporting) {
					if(Utility.GUIButton("Export Data", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(ExportData());
					}
				} else {
					EditorGUILayout.LabelField("Asset: " + Editor.GetAsset().GetName());
					EditorGUILayout.LabelField("Index: " + (Editor.GetAssetIndex()+1) + " / " + Assets.Length);
					EditorGUILayout.LabelField("Mirror: " + Editor.Mirror);
					EditorGUILayout.LabelField("Frames Per Second: " + Performance.ToString("F3"));
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, (float)(Editor.GetAssetIndex()+1) / (float)Assets.Length * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Opacity(0.75f));
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Progress * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Opacity(0.75f));

					EditorGUI.BeginDisabledGroup(Aborting);
					if(Utility.GUIButton(Aborting ? "Aborting" : "Stop", Aborting ? UltiDraw.Gold : UltiDraw.DarkRed, UltiDraw.White)) {
						Aborting = true;
					}
					EditorGUI.EndDisabledGroup();
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();

						EditorGUILayout.LabelField("Page", GUILayout.Width(40f));
						EditorGUI.BeginChangeCheck();
						int page = EditorGUILayout.IntField(Page, GUILayout.Width(40f));
						if(EditorGUI.EndChangeCheck()) {
							LoadPage(page);
						}
						EditorGUILayout.LabelField("/" + GetPages());
						
						EditorGUILayout.LabelField("Filter", GUILayout.Width(40f));
						EditorGUI.BeginChangeCheck();
						string filter = EditorGUILayout.TextField(Filter, GUILayout.Width(200f));
						if(EditorGUI.EndChangeCheck()) {
							ApplyFilter(filter);
						}

						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							foreach(Asset a in Assets) {
								a.Selected = true;
							}
						}
						if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							foreach(Asset a in Assets) {
								a.Selected = false;
							}
						}
						if(Utility.GUIButton("Current", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							string guid = Utility.GetAssetGUID(Editor.GetAsset());
							foreach(Asset a in Assets) {
								a.Selected = a.GUID == guid;
							}
						}
						EditorGUILayout.EndHorizontal();

						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							LoadPage(Mathf.Max(Page-1, 1));
						}
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							LoadPage(Mathf.Min(Page+1, GetPages()));
						}
						EditorGUILayout.EndHorizontal();
					}
					
					int start = GetStart();
					int end = GetEnd();
					for(int i=start; i<end; i++) {
						if(Instances[i].Exported) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else if(Instances[i].Selected) {
							Utility.SetGUIColor(UltiDraw.Gold);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Instances[i].Selected = EditorGUILayout.Toggle(Instances[i].Selected, GUILayout.Width(20f));
							EditorGUILayout.LabelField(Utility.GetAssetName(Instances[i].GUID));
							EditorGUILayout.EndHorizontal();
						}
					}
				}
			}
		}

		EditorGUILayout.EndScrollView();
	}

	public class Data {
		public StreamWriter File, Norm, Labels;

		public RunningStatistics[] Statistics = null;

		private Queue<float[]> Buffer = new Queue<float[]>();
		private Task Writer = null;

		private float[] Values = new float[0];
		private string[] Names = new string[0];
		private float[] Weights = new float[0];
		private int Dim = 0;

		private bool Finished = false;
		private bool Setup = false;

		public Data(StreamWriter file, StreamWriter norm, StreamWriter labels) {
			File = file;
			Norm = norm;
			Labels = labels;
			Writer = Task.Factory.StartNew(() => WriteData());
		}

		public void Feed(float value, string name, float weight=1f) {
			if(!Setup) {
				ArrayExtensions.Append(ref Values, value);
				ArrayExtensions.Append(ref Names, name);
				ArrayExtensions.Append(ref Weights, weight);
			} else {
				Dim += 1;
				Values[Dim-1] = value;
			}
		}

		public void Feed(float[] values, string name, float weight=1f) {
			for(int i=0; i<values.Length; i++) {
				Feed(values[i], name + (i+1), weight);
			}
		}

		public void Feed(bool[] values, string name, float weight=1f) {
			for(int i=0; i<values.Length; i++) {
				Feed(values[i] ? 1f : 0f, name + (i+1), weight);
			}
		}

		public void Feed(float[,] values, string name, float weight=1f) {
			for(int i=0; i<values.GetLength(0); i++) {
				for(int j=0; j<values.GetLength(1); j++) {
					Feed(values[i,j], name+(i*values.GetLength(1)+j+1), weight);
				}
			}
		}

		public void Feed(bool[,] values, string name, float weight=1f) {
			for(int i=0; i<values.GetLength(0); i++) {
				for(int j=0; j<values.GetLength(1); j++) {
					Feed(values[i,j] ? 1f : 0f, name+(i*values.GetLength(1)+j+1), weight);
				}
			}
		}

		public void Feed(Vector2 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.y, name+"Y", weight);
		}

		public void Feed(Vector3 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.y, name+"Y", weight);
			Feed(value.z, name+"Z", weight);
		}

		public void FeedXY(Vector3 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.y, name+"Y", weight);
		}

		public void FeedXZ(Vector3 value, string name, float weight=1f) {
			Feed(value.x, name+"X", weight);
			Feed(value.z, name+"Z", weight);
		}

		public void FeedYZ(Vector3 value, string name, float weight=1f) {
			Feed(value.y, name+"Y", weight);
			Feed(value.z, name+"Z", weight);
		}

		private void WriteData() {
			while(Exporting && (!Finished || Buffer.Count > 0)) {
				if(Buffer.Count > 0) {
					float[] item;
					lock(Buffer) {
						item = Buffer.Dequeue();	
					}
					//Update Mean and Std
					for(int i=0; i<item.Length; i++) {
						Statistics[i].Add(item[i]);
					}
					//Write to File
					File.WriteLine(String.Join(Separator, Array.ConvertAll(item, x => x.ToString(Accuracy))));
				} else {
					Thread.Sleep(1);
				}
			}
		}

		public void Store() {
			if(!Setup) {
				//Setup Mean and Std
				Statistics = new RunningStatistics[Values.Length];
				for(int i=0; i<Statistics.Length; i++) {
					Statistics[i] = new RunningStatistics();
				}

				//Write Labels
				for(int i=0; i<Names.Length; i++) {
					Labels.WriteLine("[" + i + "]" + " " + Names[i]);
				}
				Labels.Close();

				Setup = true;
			}

			//Enqueue Sample
			float[] item = (float[])Values.Clone();
			lock(Buffer) {
				Buffer.Enqueue(item);
			}

			//Reset Running Index
			Dim = 0;
		}

		public void Finish() {
			Finished = true;

			Task.WaitAll(Writer);

			File.Close();

			if(Setup) {
				//Write Mean
				float[] mean = new float[Statistics.Length];
				for(int i=0; i<mean.Length; i++) {
					mean[i] = Statistics[i].Mean();
				}
				Norm.WriteLine(String.Join(Separator, Array.ConvertAll(mean, x => x.ToString(Accuracy))));

				//Write Std
				float[] std = new float[Statistics.Length];
				for(int i=0; i<std.Length; i++) {
					std[i] = Statistics[i].Std();
				}
				std.Replace(0f, 1f);
				Norm.WriteLine(String.Join(Separator, Array.ConvertAll(std, x => x.ToString(Accuracy))));
			}

			Norm.Close();

			Debug.Log("输入/输出 features 数目："+Values.Length.ToString());
		}
	}

	private IEnumerator ExportData() {
		if(Editor == null) {
			Debug.Log("No editor found.");
		} else if(!System.IO.Directory.Exists(GetExportPath())) {
			Debug.Log("No export folder found at " + GetExportPath() + ".");
		} else {
			Aborting = false;
			Exporting = true;
			Progress = 0f;

			int sequence = 0;
			int items = 0;
			int samples = 0;
			DateTime timestamp = Utility.GetTimestamp();

			StreamWriter S = CreateFile("Sequences");
			Data X = new Data(CreateFile("Input"), CreateFile("InputNorm"), CreateFile("InputLabels"));
			Data Y = new Data(CreateFile("Output"), CreateFile("OutputNorm"), CreateFile("OutputLabels"));
			StreamWriter CreateFile(string name) {
				return File.CreateText(GetExportPath() + "/" + name + ".txt");
			}

			for(int i=0; i<Assets.Length; i++) {
				Assets[i].Exported = false;
			}
			for(int i=0; i<Assets.Length; i++) {
				if(Aborting) {
					break;
				}
				if(Assets[i].Selected) {
					for(int m=1; m<=2; m++) {
						if(WriteMirror==false) {
							m=2;
							Editor.SetMirror(false);
							Editor2.SetMirror(false);
						}
						else {
							if(m==1) {
								Editor.SetMirror(false);
								Editor2.SetMirror(false);
							}
							else {
								Editor.SetMirror(true);
								Editor2.SetMirror(true);
							}
						}

						MotionData data = Editor.LoadData(Assets[i].GUID); //这里会在下拉菜单中自动选中asset[?]
						MotionData data2 = Editor2.LoadData(Assets[i+1].GUID); 
						while(!data.GetScene().isLoaded) {
							Debug.Log("Waiting for scene being loaded...");
							yield return new WaitForSeconds(0f);
						}
						if(!data.Export) {
							Debug.Log("Skipping Asset: " + data.GetName());
							yield return new WaitForSeconds(0f);
							continue;
						}

						Debug.Log("Exporting asset " + data.GetName() + " " + (Editor.Mirror ? "[Mirror]" : "[Default Not Mirror]"));
						for(int r=1; r<=repeat; r++) {
							for(int shift=0; shift<=FrameShifts; shift++) {
								foreach(Sequence seq in data.Sequences) { // sequences是每个asset中手动定义的分段，截断掉没用的数据
									sequence += 1;
									float start = Editor.CeilToTargetTime(data.GetFrame(seq.Start).Timestamp);
									float end = Editor.FloorToTargetTime(data.GetFrame(seq.End).Timestamp);
									int index = 0;
									while(start + (index+1)/Editor.TargetFramerate + shift/data.Framerate <= end) {
										if(OnlyFirstXFrames>0 & samples>=OnlyFirstXFrames) {break;}

										Editor.SetRandomSeed(Editor.GetCurrentFrame().Index); 
										Editor2.SetRandomSeed(Editor2.GetCurrentFrame().Index); 
										S.WriteLine(sequence.ToString()); // 文件太大


										float tCurrent = start + index/Editor.TargetFramerate + shift/data.Framerate;
										float tNext = start + (index+1)/Editor.TargetFramerate + shift/data.Framerate;
										float[] tFutures = new float[5];
										for(int delta=2; delta<2+5; delta++) {
											tFutures[delta-2] = start + (index+delta)/Editor.TargetFramerate + shift/data.Framerate;
										}

										if(Pipeline == PIPELINE.Basketball) {
											BasketballSetup.Export(this, X, Y, tCurrent, tNext, tFutures);
										}

										X.Store();
										Y.Store();

										index += 1;
										Progress = (index/Editor.TargetFramerate) / (end-start);
										items += 1;
										samples += 1;
										if(items >= FrameBuffer) {
											Performance = items / (float)Utility.GetElapsedTime(timestamp);
											timestamp = Utility.GetTimestamp();
											items = 0;
											yield return new WaitForSeconds(0f);
										}
									}
									Progress = 0f;
								}
							}
						}
					}
					Assets[i].Exported = true;
				}
			}
			S.Close(); // 文件太大
			X.Finish();
			Y.Finish();
			

			Aborting = false;
			Exporting = false;
			Progress = 0f;
			foreach(Asset a in Assets) {
				a.Exported = false;
			}
			yield return new WaitForSeconds(0f);

			Debug.Log("Exported " + samples + " samples.");
		}
	}

	public class BasketballSetup {
		public static void Export(MotionExporter exporter, Data X, Data Y, float tCurrent, float tNext, float[] tFutures) {
			Container current = new Container(exporter.Editor, tCurrent);
			Container next = new Container(exporter.Editor, tNext);
			Container current2 = new Container(exporter.Editor2, tCurrent); 
			Container next2 = new Container(exporter.Editor2, tNext);

			// for future pose
			Container[] futures2 = new Container[5];
			for(int delta=2; delta<2+5; delta++) {
				futures2[delta-2] = new Container(exporter.Editor2, tFutures[delta-2]);
			}

			if(current.Frame.Index == next.Frame.Index) {
				Debug.LogError("Same frames for input output pairs selected!");
			}

			int SeriesPivot = next.TimeSeries.Pivot;
			int SeriesLength = next.TimeSeries.Samples.Length;

			// correct twist
			if(exporter.correct_twist==true) {
				current = CorrectTwist(exporter, current);
				current2 = CorrectTwist(exporter, current2);
				next = CorrectTwist(exporter, next);
				next2 = CorrectTwist(exporter, next2);
			}

			// Export_DualIntent(X, Y, current, current2, next, next2, SeriesPivot, SeriesLength, exporter);
			Export_MotionMatching_1(X, Y, current, current2, next, next2, SeriesPivot, SeriesLength, exporter);

			//* ground truth
			// Export_GroundTruth(X, Y, current, current2, next, next2, SeriesPivot, SeriesLength, exporter);
		}

		private static void Export_DualIntent(Data X, Data Y, Container current, Container current2, Container next, Container next2, int SeriesPivot, int SeriesLength, MotionExporter exporter) {
			//* Input
			// Character 1
			RootTrajectory(X, next, current, 0, SeriesLength);
			JointTrajectory(X, next, current, 0, SeriesPivot);
			Pose(X, current, current, exporter);
			// Character1 in char2 space
			RootTrajectory(X, next, current2, 0, SeriesLength);
			JointTrajectory(X, next, current2, 0, SeriesPivot);
			Pose(X, current, current2, exporter);
			// Character 2
			RootTrajectory(X, next2, current2, 0, SeriesPivot);
			JointTrajectory(X, next2, current2, 0, SeriesPivot);
			Pose(X, current2, current2, exporter);

			//* Output 
			// Character 1
			JointTrajectory(Y, next, next2, SeriesPivot+1, SeriesLength);
			// Character 2
			RootUpdate(Y, next2, current2);
			RootTrajectory(Y, next2, next2, SeriesPivot+1, SeriesLength);
			JointTrajectory(Y, next2, next2, SeriesPivot+1, SeriesLength);
			Pose(Y, next2, next2, exporter);
		}

		private static void Export_MotionMatching_1(Data X, Data Y, Container current, Container current2, Container next, Container next2, int SeriesPivot, int SeriesLength, MotionExporter exporter) {
			// Input
			RootTrajectory(X, next, current, SeriesPivot+1, SeriesLength);
			Pose(X, current, current, exporter);

			// Output 
			RootUpdate(Y, next, current);
			RootTrajectory(Y, next, next, SeriesPivot+1, SeriesLength);
			Pose(Y, next, next, exporter);
			JointTrajectory(Y, next, next, SeriesPivot+1, SeriesLength);
		}

		private static void Export_GroundTruth(Data X, Data Y, Container current, Container current2, Container next, Container next2, int SeriesPivot, int SeriesLength, MotionExporter exporter) {
			// Character 1 
			RootUpdate(Y, next, current);
			GlobalRoot(Y, next);
			RootTrajectory(Y, next, next, SeriesPivot+1, SeriesLength);
			Pose(Y, next, next, exporter);
			// Character 2
			RootUpdate(Y, next2, current2);
			GlobalRoot(Y, next2);
			RootTrajectory(Y, next2, next2, SeriesPivot+1, SeriesLength);
			Pose(Y, next2, next2, exporter);
		}

		private static void RootTrajectory(Data Data, Container From, Container To, int Start, int End) {
			for(int k=Start; k<End; k++) { 
				Data.FeedXZ(From.RootSeries.GetPosition(k).GetRelativePositionTo(To.Root), "TrajectoryPosition"+(k+1));
				Data.FeedXZ(From.RootSeries.GetDirection(k).GetRelativeDirectionTo(To.Root), "TrajectoryDirection"+(k+1));
			}
		}

		private static void JointTrajectory(Data Data, Container From, Container To, int StartKey, int EndKey) {
			for(int k=StartKey; k<EndKey; k++) { 
				Data.Feed(From.HeadSeries.GetPosition(k).GetRelativePositionTo(To.Root), "HeadTrajectoryPosition"+(k+1));
				Data.Feed(From.HeadSeries.GetDirection(k).GetRelativeDirectionTo(To.Root), "HeadTrajectoryDirection"+(k+1));
			}
			for(int k=StartKey; k<EndKey; k++) { 
				Data.Feed(From.LeftHandSeries.GetPosition(k).GetRelativePositionTo(To.Root), "LeftHandTrajectoryPosition"+(k+1));
				Data.Feed(From.LeftHandSeries.GetDirection(k).GetRelativeDirectionTo(To.Root), "LeftHandTrajectoryDirection"+(k+1));
			}
			for(int k=StartKey; k<EndKey; k++) { 
				Data.Feed(From.RightHandSeries.GetPosition(k).GetRelativePositionTo(To.Root), "RightHandTrajectoryPosition"+(k+1));
				Data.Feed(From.RightHandSeries.GetDirection(k).GetRelativeDirectionTo(To.Root), "RightHandTrajectoryDirection"+(k+1));
			}
			for(int k=StartKey; k<EndKey; k++) { 
				Data.Feed(From.LeftLegSeries.GetPosition(k).GetRelativePositionTo(To.Root), "LeftLegTrajectoryPosition"+(k+1));
				Data.Feed(From.LeftLegSeries.GetDirection(k).GetRelativeDirectionTo(To.Root), "LeftLegTrajectoryDirection"+(k+1));
			}
			for(int k=StartKey; k<EndKey; k++) { 
				Data.Feed(From.RightLegSeries.GetPosition(k).GetRelativePositionTo(To.Root), "RightLegTrajectoryPosition"+(k+1));
				Data.Feed(From.RightLegSeries.GetDirection(k).GetRelativeDirectionTo(To.Root), "RightLegTrajectoryDirection"+(k+1));
			}
		}

		private static void Pose(Data Data, Container From, Container To, MotionExporter Exporter) {
			for(int k=0; k<From.ActorPosture.Length; k++) {
				Data.Feed(From.ActorPosture[k].GetPosition().GetRelativePositionTo(To.Root), "Bone"+(k+1)+Exporter.Editor.GetActor().Bones[k].GetName()+"Position");
				Data.Feed(From.ActorPosture[k].GetForward().GetRelativeDirectionTo(To.Root), "Bone"+(k+1)+Exporter.Editor.GetActor().Bones[k].GetName()+"Forward");
				Data.Feed(From.ActorPosture[k].GetUp().GetRelativeDirectionTo(To.Root), "Bone"+(k+1)+Exporter.Editor.GetActor().Bones[k].GetName()+"Up");
			}
		}

		private static void RootUpdate(Data Data, Container From, Container To) {
			Matrix4x4 delta = From.Root.GetRelativeTransformationTo(To.Root);
			Data.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
		}

		private static void GlobalRoot(Data Data, Container Container) {
			Data.Feed(Container.Root.GetPosition(), "RootPosition");
			Data.Feed(Container.Root.GetForward(), "RootForward");
		}

		private static void FuturePoses() {
			// for(int delta=2; delta<2+5; delta++) {
			// 	Matrix4x4 root2_future = futures2[delta-2].Root;
			// 	Matrix4x4 delta2_future = root2_future.GetRelativeTransformationTo(current2.Root);
			// 	Y.Feed(new Vector3(delta2_future.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta2_future.GetForward(), Vector3.up), delta2_future.GetPosition().z), "RootUpdate2_Future");

			// 	for(int k=0; k<futures2[delta-2].ActorPosture.Length; k++) {
			// 		Y.Feed(futures2[delta-2].ActorPosture[k].GetPosition().GetRelativePositionTo(root2_future), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Position2_Future"+delta.ToString());
			// 		Y.Feed(futures2[delta-2].ActorPosture[k].GetForward().GetRelativeDirectionTo(root2_future), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Forward2_Future"+delta.ToString());
			// 		Y.Feed(futures2[delta-2].ActorPosture[k].GetUp().GetRelativeDirectionTo(root2_future), "Bone"+(k+1)+exporter.Editor.GetActor().Bones[k].GetName()+"Up2_Future"+delta.ToString());
			// 	}
			// }
		}

		private static Container CorrectTwist(MotionExporter exporter, Container character) {
			for(int k=0; k<character.ActorPosture.Length; k++) {
				Vector3 position;
				Quaternion rotation;
				position = character.ActorPosture[k].GetPosition();
				rotation = character.ActorPosture[k].GetRotation();

				if(exporter.Editor.GetActor().Bones[k].Childs.Length==1) {
					int childIndex = exporter.Editor.GetActor().Bones[k].GetChild(0).Index;
					Vector3 childPosition = character.ActorPosture[childIndex].GetPosition();
					Vector3 aligned = (position - childPosition).normalized;
					rotation = Quaternion.FromToRotation(rotation.GetRight(), aligned) * rotation;
				}
				character.ActorPosture[k] = Matrix4x4.TRS(position, rotation, Vector3.one);
			}
			
			return character;
		}

		private class Container {
			public MotionData Asset;
			public Frame Frame;

			public TimeSeries TimeSeries;
			public RootSeries RootSeries;
			public DribbleSeries DribbleSeries;
			public StyleSeries StyleSeries;
			public ContactSeries ContactSeries;
			public PhaseSeries PhaseSeries;

			//* joint trajectory
			public RootSeries HeadSeries;
			public RootSeries LeftHandSeries;
			public RootSeries RightHandSeries;
			public RootSeries LeftLegSeries;
			public RootSeries RightLegSeries;

			//Actor Features
			public Matrix4x4 Root;
			public Matrix4x4[] ActorPosture;
			public Vector3[] ActorVelocities;
			public float[] RivalBoneDistances;

			public Container(MotionEditor editor, float timestamp) {
				editor.LoadFrame(timestamp);
				Asset = editor.GetAsset();
				Frame = editor.GetCurrentFrame();
				TimeSeries = editor.GetTimeSeries();
				RootSeries = (RootSeries)Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				// ContactSeries = (ContactSeries)Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);

				//* joint trajectory
				HeadSeries = (RootSeries)Asset.GetModule<HeadModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				LeftHandSeries = (RootSeries)Asset.GetModule<LeftHandModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				RightHandSeries = (RootSeries)Asset.GetModule<RightHandModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				LeftLegSeries = (RootSeries)Asset.GetModule<LeftLegModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);
				RightLegSeries = (RootSeries)Asset.GetModule<RightLegModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror);

				// Root = editor.GetActor().transform.GetWorldMatrix(true);
				Root = RootSeries.GetTransformation(TimeSeries.PivotIndex); //todo
				ActorPosture = editor.GetActor().GetBoneTransformations();
				ActorVelocities = editor.GetActor().GetBoneVelocities();
				// RivalBoneDistances = DribbleSeries.GetInteractorBoneDistances();
			}
		}

		
	}
}
#endif