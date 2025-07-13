using UnityEngine;
using System.Collections.Generic;
using System;

public class State {
    public Matrix4x4 CurrentRoot; 
    public Matrix4x4 UpdatedRoot;
    public Vector3 Delta;
    public Vector3[] Positions; 
    public Vector3[] Forwards;
    public Vector3[] Upwards;
    public float Distance;
    public RootSeries RootSeries;
    public Vector3 ControlDirection;
    public float Angle;
    public RootSeries HeadSeries;
    public RootSeries LeftHandSeries;
    public RootSeries RightHandSeries;
    public RootSeries LeftLegSeries;
    public RootSeries RightLegSeries;

    public RootSeries HeadSeries2;
    public RootSeries LeftHandSeries2;
    public RootSeries RightHandSeries2;
    public RootSeries LeftLegSeries2;
    public RootSeries RightLegSeries2;

    public State(Actor character, Matrix4x4 updatedRoot, Vector3[] positions, Vector3[] forwards, Vector3[] ups, RootSeries rootSeries, GameObject arrow, int controlFeature, RootSeries headSeries, RootSeries leftHandSeries, RootSeries rightHandSeries, RootSeries leftLegSeries, RootSeries rightLegSeries, RootSeries headSeries2, RootSeries leftHandSeries2, RootSeries rightHandSeries2, RootSeries leftLegSeries2, RootSeries rightLegSeries2) {
        CurrentRoot = character.GetRoot().GetWorldMatrix(true); // deep copy
        UpdatedRoot = updatedRoot; // deep copy

        Positions = (Vector3[])positions.Clone();
        Forwards = (Vector3[])forwards.Clone();
        Upwards = (Vector3[])ups.Clone();

        RootSeries = RootSeries.CloneSeries(rootSeries);

        HeadSeries = RootSeries.CloneSeries(headSeries);
        LeftHandSeries = RootSeries.CloneSeries(leftHandSeries);
        RightHandSeries = RootSeries.CloneSeries(rightHandSeries);
        LeftLegSeries = RootSeries.CloneSeries(leftLegSeries);
        RightLegSeries = RootSeries.CloneSeries(rightLegSeries);

        HeadSeries2 = RootSeries.CloneSeries(headSeries2);
        LeftHandSeries2 = RootSeries.CloneSeries(leftHandSeries2);
        RightHandSeries2 = RootSeries.CloneSeries(rightHandSeries2);
        LeftLegSeries2 = RootSeries.CloneSeries(leftLegSeries2);
        RightLegSeries2 = RootSeries.CloneSeries(rightLegSeries2);

        if(controlFeature==0 || arrow.activeSelf==false) { // control by pose distance or control signal is unavailable
            ComputeDistance(character);
        }
        else if(controlFeature==1) { // control by angles
            ControlDirection = arrow.transform.rotation.GetForward();
            ComputeAngle();
        }
    }
    public void ComputeDistance(Actor character) {
        Distance = 0f;
        for(int i=0; i<Positions.Length; i++) {
            // Distance += Mathf.Pow(Vector3.Distance(character.Bones[i].Transform.position.GetRelativePositionTo(CurrentRoot), Positions[i]), 2f);
            Distance += Mathf.Pow(Vector3.Distance(character.Bones[i].Transform.position, Positions[i].GetRelativePositionFrom(UpdatedRoot)), 2f);
            // Distance += Mathf.Pow(Vector3.Distance(character.transform.GetComponent<Char2Controller>().Opponent.transform.position, Positions[i].GetRelativePositionFrom(UpdatedRoot)), 2f);
        }

        // for(int i=RootSeries.Pivot; i<RootSeries.Samples.Length; i++) {
        //     // Distance += (RootSeries.GetPosition(i) - actor.transform.position).magnitude;
        //     Distance += (RootSeries.GetPosition(i) - character.transform.GetComponent<Char2Controller>().Opponent.transform.GetComponent<Char1Controller>().RootSeries.GetPosition(i)).magnitude;
        //     // Distance += Vector3.Angle((actor.transform.position - RootSeries.GetPosition(i)), RootSeries.GetRotation(i).GetForward());
        // }

        // Distance += Vector3.Angle(character.transform.GetComponent<Char2Controller>().Opponent.transform.position - UpdatedRoot.GetPosition(), UpdatedRoot.GetForward());
        // Distance += (character.transform.GetComponent<Char2Controller>().GT.GetRoot().GetWorldMatrix(true).GetPosition() - UpdatedRoot.GetPosition()).magnitude;
    }

    public void ComputeAngle() {
        Distance = 0f;
        Vector3 updatedDirection = UpdatedRoot.GetPosition() - CurrentRoot.GetPosition();
        Distance = Vector3.Angle(updatedDirection, ControlDirection);
        // Debug.Log(Distance);
    }

    public void DrawSamples(Actor actor, Camera camera, bool drawPlayer1, bool drawPlayer2) {
        for(int i=0; i<actor.Bones.Length; i++) {
            actor.Bones[i].Transform.position = Positions[i];
            actor.Bones[i].Transform.rotation = Quaternion.LookRotation(Forwards[i], Upwards[i]);
        }

        Color redSkeleton = new Color(0.74f, 0.24f, 0.33f, 1f);
        Color orange = UltiDraw.Orange.Opacity(0.5f);
		Color red = UltiDraw.Red.Opacity(0.5f);

		RootSeries.Draw(camera, redSkeleton, true, drawHalf: 2);

        Color jointSeriesColor = orange;
        if(drawPlayer1) {
            HeadSeries.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            LeftHandSeries.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            LeftLegSeries.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            RightHandSeries.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            RightLegSeries.Draw(camera, jointSeriesColor, false, drawHalf: 2);
        }
        if(drawPlayer2) {
            HeadSeries2.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            LeftHandSeries2.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            LeftLegSeries2.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            RightHandSeries2.Draw(camera, jointSeriesColor, false, drawHalf: 2);
            RightLegSeries2.Draw(camera, jointSeriesColor, false, drawHalf: 2);
        }
    }


    private Matrix4x4[] GetTransformations() {
        Matrix4x4[] m = new Matrix4x4[Positions.Length];
        for(int i=0; i<m.Length; i++) {
            m[i] = Matrix4x4.TRS(Positions[i], Quaternion.LookRotation(Forwards[i], Upwards[i]), Vector3.one);
        }
        return m;
    }
}