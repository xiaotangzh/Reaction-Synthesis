using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrajectoryCameraController : MonoBehaviour
{
    public GameObject target;
    public MotionController motionController;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = new Vector3(target.transform.position.x, transform.position.y, target.transform.position.z);
        transform.rotation = Quaternion.LookRotation(transform.rotation.GetForward(), target.transform.rotation.GetForward());

        // if(motionController.countFrame>1) {
        //     TimeSeries timeSeries = motionController.timeSeries;
        //     RootSeries rootSeries = motionController.rootSeries;
        //     Vector3 direction = rootSeries.Transformations[timeSeries.PivotIndex+1].GetPosition() - rootSeries.Transformations[timeSeries.PivotIndex].GetPosition();
        //     transform.rotation = Quaternion.LookRotation(transform.rotation.GetForward(), direction);
        // }
    }
}
