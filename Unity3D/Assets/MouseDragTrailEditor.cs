// using UnityEngine;
// using UnityEditor;
// using System.Collections.Generic;
// // using Accord.Math;
// using System;

// [InitializeOnLoad]
// public class MouseDragTrailEditor
// {
//     private static bool isRecording = false;
//     private static List<Vector3> trailPoints = new List<Vector3>();

//     // Define the plane you want to project onto. For instance, this uses the XZ plane (Y = 0).
//     private static Plane projectionPlane = new Plane(Vector3.up, Vector3.zero);
//     public static List<Matrix4x4> Trajectory;

//     static MouseDragTrailEditor()
//     {
//         SceneView.duringSceneGui += TrackMouseDrag;
//     }


//     static void TrackMouseDrag(SceneView sceneView)
//     {
//         Event e = Event.current;

//         // 清空trajectory
//         // if (e.type == EventType.KeyDown && e.keyCode == KeyCode.Space && e.control) // Check if Ctrl is pressed using e.control
//         // {
//         //     trailPoints.Clear();
//         //     Array.Clear(Trajectory, 0, Trajectory.Length);
//         //     return; // Return here to prevent toggling the isRecording state when Ctrl+Space is pressed.
//         // }

//         // Start recording on Space key press.
//         if (e.type == EventType.KeyDown && e.keyCode == KeyCode.Space)
//         {
//             isRecording = !isRecording;  // Toggle recording state
//             if (!isRecording)
//             {
//                 // trailPoints.Clear();
//                 // Debug.Log(trailPoints.Count);
//                 Trajectory = TransformToMatrix4x4(trailPoints);
//                 // Debug.Log(Trajectory.Length);
//             }
//         }

//         if (isRecording && e.type == EventType.MouseMove)  // Only record when moving, not dragging
//         {
//             Ray ray = HandleUtility.GUIPointToWorldRay(e.mousePosition);
//             if (projectionPlane.Raycast(ray, out float enter))
//             {
//                 Vector3 hitPoint = ray.GetPoint(enter);
//                 trailPoints.Add(hitPoint);
//             }
//         }

//         // Draw the trail in the scene view for visual feedback.
//         if (trailPoints.Count > 1)
//         {
//             for (int i = 0; i < trailPoints.Count - 1; i++)
//             {
//                 Handles.DrawLine(trailPoints[i], trailPoints[i + 1]);
//             }
//             sceneView.Repaint(); // Make sure the Scene view updates to show the lines.
//         }

//         if(!isRecording && Trajectory!=null) {
//             DrawTrajectory(Trajectory);
//         }
//     }

//     static List<Matrix4x4> TransformToMatrix4x4(List<Vector3> trailPoints) {
//         //将第一个点设置为0
//         // trailPoints[0] = new Vector3(0,0,0);

//         // interpolate
//         List<Vector3> interpolated = new List<Vector3>();
//         for(int i=0; i<trailPoints.Count-1; i+=1) {
//             Vector3 interpolate = (trailPoints[i]+trailPoints[i+1])/2;
//             interpolated.Insert(i, trailPoints[i]);
//             interpolated.Insert(i+1, interpolate);
//         }
//         trailPoints = interpolated;

//         // downsample
//         List<Vector3> downsampled = new List<Vector3>();
//         int rate = 1; //downsample的比率
//         for(int i=0; i<trailPoints.Count; i+=rate) {
//             downsampled.Add(trailPoints[i]);
//         }

//         // 转换成matrix4x4
//         List<Matrix4x4> M = new List<Matrix4x4>();
//         for(int i=0; i<downsampled.Count-1; i++){
//             // Matrix4x4 m = Matrix4x4.TRS(downsampled[i], Quaternion.identity, Vector3.one); // 只有pos
//             Matrix4x4 m = Matrix4x4.TRS(downsampled[i], Quaternion.LookRotation(downsampled[i+1]-downsampled[i]), Vector3.one);
//             M.Add(m);
//         }

//         return M;
//     }

//     static void DrawTrajectory(List<Matrix4x4> traj) {
//         UltiDraw.Begin();
//         for(int i=0; i<traj.Count-1; i+=1) {
//             UltiDraw.DrawLine(traj[i].GetPosition(), traj[i+1].GetPosition(), traj[i].GetUp(), 0.1f, UltiDraw.Green);
//         }
//         for(int i=0; i<traj.Count; i+=1) {
//             UltiDraw.DrawCircle(traj[i].GetPosition(), 0.05f, UltiDraw.Black);
//         }
//         UltiDraw.End();
//     }

//     public static Matrix4x4[] GetTrajectoryArray() {
//         if(Trajectory==null) {return null;}
//         else{return Trajectory.ToArray();}
//     }
// }
