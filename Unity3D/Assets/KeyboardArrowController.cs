using UnityEngine;
using UnityEngine.UI;

public class KeyboardArrowController : MonoBehaviour
{
    public Sprite newSprite;

    private Image up;
    private Image down;
    private Image left;
    private Image right;

    private Sprite original;

    void Start()
    {
        up = transform.Find("Up").GetComponent<Image>();
        down = transform.Find("Down").GetComponent<Image>();
        left = transform.Find("Left").GetComponent<Image>();
        right = transform.Find("Right").GetComponent<Image>();
        original = up.sprite;
    }

    void Update()
    {
        // Up (W or UpArrow)
        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
        {
            up.sprite = newSprite;
        }
        else
        {
            up.sprite = original;
        }

        // Down (S or DownArrow)
        if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
        {
            down.sprite = newSprite;
        }
        else
        {
            down.sprite = original;
        }

        // Left (A or LeftArrow)
        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
        {
            left.sprite = newSprite;
        }
        else
        {
            left.sprite = original;
        }

        // Right (D or RightArrow)
        if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
        {
            right.sprite = newSprite;
        }
        else
        {
            right.sprite = original;
        }
    }
}
