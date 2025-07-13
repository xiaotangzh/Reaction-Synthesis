using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEditor;
using System;

public class SceneInputHandler : EditorWindow
{
    private GameObject targetObject;  // GameObject引用
    private Vector3 objectPosition;
    public Matrix4x4[] predefinedTrajectory = new Matrix4x4[0];

    [MenuItem("Window/Scene Input Handler")]
    public static void ShowWindow()
    {
        EditorWindow.GetWindow(typeof(SceneInputHandler));
    }

    private void OnEnable()
    {
        SceneView.duringSceneGui += SceneGUI;
    }

    private void OnDisable()
    {
        SceneView.duringSceneGui -= SceneGUI;
    }

    void SceneGUI(SceneView sceneView)
    {
        Event e = Event.current;
        switch (e.type)
        {
            case EventType.KeyDown:  // 监听键盘按键按下事件
                if (e.keyCode == KeyCode.Space)  // 如果是空格键
                {
                    Debug.Log("Space Key Pressed in Scene!");

                    if (targetObject != null)  // 如果GameObject存在
                    {
                        objectPosition = targetObject.transform.position;
                        Debug.Log($"Position of {targetObject.name}: {objectPosition}");
                        Matrix4x4 m = Matrix4x4.TRS(objectPosition, Quaternion.identity, Vector3.one);

                        if(predefinedTrajectory.Length==0) {
                            predefinedTrajectory = new Matrix4x4[1];
                            predefinedTrajectory[0] = m;
                        }
                        else {
                            Matrix4x4[] _predefinedTrajectory = new Matrix4x4[predefinedTrajectory.Length+1];
                            Array.Copy(predefinedTrajectory, _predefinedTrajectory, predefinedTrajectory.Length);
                            _predefinedTrajectory[_predefinedTrajectory.Length-1] = m;
                            predefinedTrajectory = _predefinedTrajectory;
                        }
                        Debug.Log("Trajectory长度："+predefinedTrajectory.Length.ToString());
                    }
                }
                break;
            
            // ... You can handle other mouse or keyboard events here as needed.
        }

        UltiDraw.Begin();
        for(int i=0; i<predefinedTrajectory.Length-1; i+=1) {
                UltiDraw.DrawLine(predefinedTrajectory[i].GetPosition(), predefinedTrajectory[i+1].GetPosition(), predefinedTrajectory[i].GetUp(), 0.1f, UltiDraw.Green);
        }
        UltiDraw.End();
    }

    // 添加一个绘制窗口的方法，以便用户可以在编辑器窗口中选择GameObject
    private void OnGUI()
    {
        targetObject = EditorGUILayout.ObjectField("Target Object:", targetObject, typeof(GameObject), true) as GameObject;

        if (GUILayout.Button("清除Trajectory"))
        {
            predefinedTrajectory = new Matrix4x4[0]; 
            Debug.Log("清楚完毕，Trajectory长度："+predefinedTrajectory.Length.ToString());
        }

        if (GUILayout.Button("初始化位置"))
        {
            targetObject.transform.position = new Vector3(0,0,0);
        }
    }





}