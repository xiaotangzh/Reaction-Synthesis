using UnityEngine;
using System.Collections.Generic;

[CreateAssetMenu(menuName = "Mouse Trail Data")]
public class MouseTrailData : ScriptableObject
{
    public List<Vector3> trailPoints = new List<Vector3>();
}
