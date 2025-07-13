using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TextController : MonoBehaviour
{
    public Transform obj;
    public UnityEngine.UI.Text txt;
    public LineRenderer lineRenderer;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        txt.text = "up: " + (obj.rotation * Vector3.up).ToString() + "\n" + "forward: " + (obj.rotation * Vector3.forward).ToString();

        // UltiDraw.Begin();
        // UltiDraw.DrawArrow(obj.position, obj.position + obj.rotation * Vector3.forward * 100, 1, 1, 1, Color.black);
        // UltiDraw.End();

        lineRenderer.positionCount = 2; // 设置LineRenderer的顶点数为2（起点和终点）

        // 设置LineRenderer的起始和终止点
        lineRenderer.SetPosition(0, obj.position);
        lineRenderer.SetPosition(1, obj.position + obj.rotation * Vector3.forward);
        lineRenderer.startColor = Color.blue;
        lineRenderer.endColor = Color.blue;
    }


}
