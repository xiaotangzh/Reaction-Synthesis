using System.Runtime.InteropServices;
// using SharpDX;
using UnityEngine;

public class RootSeries : ComponentSeries {
    public Color defaultColor = UltiDraw.Orange.Opacity(0.75f);
    public bool DrawDirection = true;

    public Matrix4x4[] Transformations;
    public Vector3[] Velocities;
    public int drawHalf = 0;

    public bool CollisionsResolved;

    public RootSeries(TimeSeries global) : base(global) {
        Transformations = new Matrix4x4[Samples.Length];
        Velocities = new Vector3[Samples.Length];

        for(int i=0; i<Samples.Length; i++) {
            Transformations[i] = Matrix4x4.identity;
            Velocities[i] = Vector3.zero;
        }
    }

    public RootSeries(TimeSeries global, Transform transform) : base(global) {
        Transformations = new Matrix4x4[Samples.Length];
        Velocities = new Vector3[Samples.Length];
        Matrix4x4 root = transform.GetWorldMatrix(true);

        for(int i=0; i<Samples.Length; i++) {
            Transformations[i] = root;
            Velocities[i] = Vector3.zero;
        }
    }

    public static RootSeries CloneSeries(RootSeries cloneSeries) {
        RootSeries newSeries = new RootSeries(cloneSeries);
        for(int i=0; i<cloneSeries.SampleCount; i++) {
            newSeries.Transformations[i] = cloneSeries.Transformations[i];
        }
        return newSeries;
    }

    public void SetTransformation(int index, Matrix4x4 transformation) {
        Transformations[index] = transformation;
    }

    public Matrix4x4 GetTransformation(int index) {
        return Transformations[index];
    }

    public void SetPosition(int index, Vector3 position) {
        Matrix4x4Extensions.SetPosition(ref Transformations[index], position);
    }

    public Vector3 GetPosition(int index) {
        return Transformations[index].GetPosition();
    }

    public void SetRotation(int index, Quaternion rotation) {
        Matrix4x4Extensions.SetRotation(ref Transformations[index], rotation);
    }

    public Quaternion GetRotation(int index) {
        return Transformations[index].GetRotation();
    }

    public void SetDirection(int index, Vector3 direction) {
        Matrix4x4Extensions.SetRotation(ref Transformations[index], Quaternion.LookRotation(direction == Vector3.zero ? Vector3.forward : direction, Vector3.up));
    }

    public Vector3 GetDirection(int index) {
        return Transformations[index].GetForward();
    }

    public void SetVelocity(int index, Vector3 velocity) {
        Velocities[index] = velocity;
    }

    public Vector3 GetVelocity(int index) {
        return Velocities[index];
    }

    public void Translate(int index, Vector3 delta) {
        SetPosition(index, GetPosition(index) + delta);
    }

    public void Rotate(int index, Quaternion delta) {
        SetRotation(index, GetRotation(index) * delta);
    }

    public void Rotate(int index, float angles, Vector3 axis) {
        Rotate(index, Quaternion.AngleAxis(angles, axis));
    }

    public void ResolveCollisions(float safety, LayerMask mask) {
        CollisionsResolved = false;
        for(int i=Pivot; i<Samples.Length; i++) {
            Vector3 previous = GetPosition(i-1);
            Vector3 current = GetPosition(i);
            RaycastHit hit;
            if(Physics.Raycast(previous, (current-previous).normalized, out hit, Vector3.Distance(current, previous), mask)) {
                //This makes sure no point would ever fall into a geometry volume by projecting point i to i-1
                for(int j=i; j<Samples.Length; j++) {
                    SetPosition(j, GetPosition(j-1));
                }
            }
            //This generates a safety-slope around objects over multiple frames in a waterflow-fashion
            Vector3 corrected = SafetyProjection(GetPosition(i));
            if(corrected != current) {
                CollisionsResolved = true;
            }
            SetPosition(i, corrected);
            SetVelocity(i, GetVelocity(i) + (corrected-current) / (Samples[i].Timestamp - Samples[i-1].Timestamp));
        }

        Vector3 SafetyProjection(Vector3 pivot) {
            Vector3 point = Utility.GetClosestPointOverlapSphere(pivot, safety, mask);
            return point + safety * (pivot - point).normalized;
        }
    }

    public override void Increment(int start, int end) {
        for(int i=start; i<end; i++) {
            Transformations[i] = Transformations[i+1];
            Velocities[i] = Velocities[i+1];
        }
    }

    public override void Interpolate(int start, int end) {
        for(int i=start; i<end; i++) {
            float weight = (float)(i % Resolution) / (float)Resolution;
            int prevIndex = GetPreviousKey(i).Index;
            int nextIndex = GetNextKey(i).Index;
            if(prevIndex != nextIndex) {
                SetPosition(i, Vector3.Lerp(GetPosition(prevIndex), GetPosition(nextIndex), weight));
                SetDirection(i, Vector3.Lerp(GetDirection(prevIndex), GetDirection(nextIndex), weight).normalized);
                SetVelocity(i, Vector3.Lerp(GetVelocity(prevIndex), GetVelocity(nextIndex), weight));
            }
        }
    }

    public override void GUI(Camera canvas=null) {
        
    }

    public override void Draw(Camera canvas=null) {
        if(DrawScene) {
            UltiDraw.Begin(canvas);

            float size = 0.025f; //0.05f;
            float width = 0.03f;
            int step = Resolution;
            int start = 0;
            int length = Transformations.Length;

            if(drawHalf!=0) {
                if(drawHalf==1) { // first half
                    length=(int)Transformations.Length/2+1; 
                    start=0;
                }
                else { // second half
                    length=Transformations.Length;
                    start=(int)Transformations.Length/2;
                }
            }

            //Connections
            for(int i=start; i<length-step; i+=step) {
                UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Vector3.up, width, UltiDraw.Black);
            }

            //Positions
            for(int i=start; i<length; i+=step) {
                UltiDraw.DrawCircle(Transformations[i].GetPosition(), size, defaultColor);
            }

            //Directions
            if(DrawDirection) {
                for(int i=start; i<length; i+=step) {
                    UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + 0.25f*Transformations[i].GetForward(), Transformations[i].GetUp(), size, 0f, defaultColor); 
                }
            }

            // //Velocities
            // for(int i=0; i<Velocities.Length; i+=step) {
            //     UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + GetTemporalScale(Velocities[i]), Transformations[i].GetUp(), size*0.0125f, 0f, UltiDraw.DarkGreen.Opacity(0.25f));
            // }
            
            //Target
            // UltiDraw.DrawSphere(TargetPosition, Quaternion.identity, 0.25f, UltiDraw.Black);
            // UltiDraw.DrawLine(TargetPosition, TargetPosition + 0.25f*TargetDirection, Vector3.up, size*0.05f, 0f, UltiDraw.Orange);
            // UltiDraw.DrawLine(TargetPosition, TargetPosition + TargetVelocity, Vector3.up, size*0.025f, 0f, UltiDraw.DarkGreen);

            UltiDraw.End();
        }
    }



    public void Draw(Camera canvas=null, Color color=default, bool withDirection=true, int drawHalf=0) {
        if(DrawScene) {
            UltiDraw.Begin(canvas);
            
            int step = Resolution;
            int length = Transformations.Length;
            int start = 0;

            float size = 0.05f;
            float width = 0.03f;

            if(drawHalf!=0) {
                if(drawHalf==1) { // first half
                    length=(int)Transformations.Length/2; 
                    start=0;
                }
                else { // second half
                    length=Transformations.Length;
                    start=(int)Transformations.Length/2;
                }
            }

            // Connections
            for(int i=start; i<length-step; i+=step) {
                // UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Transformations[i].GetUp(), width, UltiDraw.Black);
                UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Vector3.up, width, UltiDraw.Black);
            }

            //Positions
            for(int i=start; i<length; i+=step) {
                UltiDraw.DrawCircle(Transformations[i].GetPosition(), size, color);
            }

            //Directions
            if(withDirection==true) {
                for(int i=start; i<length; i+=step) {
                    UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + 0.25f*Transformations[i].GetForward(), Transformations[i].GetUp(), size, 0f, color); 
                }
            }
            UltiDraw.End();
        }
    }
    public void Draw(Camera canvas=null, Color color=default, bool withDirection=true, int drawHalf=0, Matrix4x4 local = new Matrix4x4()) {
        if(DrawScene) {
            UltiDraw.Begin(canvas);
            
            int step = Resolution;
            int length = Transformations.Length;
            int start = 0;
            float size = 0.05f;
            float width = 0.03f;

            if(drawHalf!=0) {
                if(drawHalf==1) { // first half
                    length=(int)Transformations.Length/2; 
                    start=0;
                }
                else { // second half
                    length=Transformations.Length;
                    start=(int)Transformations.Length/2;
                }
            }

            // Connections
            for(int i=start; i<length-step; i+=step) {
                UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Vector3.up, width, UltiDraw.Black);
            }
            for(int i=start; i<length-step; i+=step) {
                UltiDraw.DrawLine(Transformations[i].GetPosition(), local.GetPosition(), Vector3.up, width, new Color(0.74f, 0.24f, 0.33f, 1f));
            }

            //Positions
            for(int i=start; i<length; i+=step) {
                UltiDraw.DrawCircle(Transformations[i].GetPosition(), size, color);
            }

            UltiDraw.End();
        }
    }

    public void DrawWithOffset(Camera canvas=null, Color color=default, bool withDirection=true, int drawHalf=0, float offset=0f) {
        if(DrawScene) {
            UltiDraw.Begin(canvas);

            float size = 0.05f;
            float width = 0.03f;
            int step = Resolution;
            int length = Transformations.Length;
            int start = 0;
            if(drawHalf!=0) {
                if(drawHalf==1) {
                    length=(int)Transformations.Length/2; 
                    start=0;
                }
                else {
                    length=Transformations.Length;
                    start=(int)Transformations.Length/2;
                }
            }

            // Connections
            for(int i=start; i<length-step; i+=step) {
                // UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Transformations[i].GetUp(), width, UltiDraw.Black);

                Vector3 tempPosition = new Vector3(Transformations[i].GetPosition().x, Transformations[i].GetPosition().y-offset, Transformations[i].GetPosition().z);
                Vector3 tempPosition2 = new Vector3(Transformations[i+step].GetPosition().x, Transformations[i+step].GetPosition().y-offset, Transformations[i+step].GetPosition().z);
                UltiDraw.DrawLine(tempPosition, tempPosition2, Vector3.up, width, UltiDraw.Black);
            }

            //Positions
            for(int i=start; i<length; i+=step) {
                Vector3 tempPosition = new Vector3(Transformations[i].GetPosition().x, Transformations[i].GetPosition().y-offset, Transformations[i].GetPosition().z);
                UltiDraw.DrawCircle(tempPosition, size, color);
            }

            //Directions
            if(withDirection==true) {
            for(int i=start; i<length; i+=step) {
                Vector3 tempPosition = new Vector3(Transformations[i].GetPosition().x, Transformations[i].GetPosition().y-offset, Transformations[i].GetPosition().z);
                UltiDraw.DrawLine(tempPosition, tempPosition + 0.25f*Transformations[i].GetForward(), Transformations[i].GetUp(), size, 0f, color); //默认size：size*0.025f
            }
            }
            UltiDraw.End();
        }
    }
}