import java.io.*;

class PyCaller {
    private static final String DATA_SWAP = "temp.txt";
    private static final String PY_URL = System.getProperty("user.dir") + "\\motor_detection.py";

    public static void writeImagePath() {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new FileWriter(new File(DATA_SWAP)));
        } catch (IOException e) {
            e.printStackTrace();
        }
        pw.close();
    }

    public static String readAnswer() {
        BufferedReader br;
        String answer = null;
        try {
            br = new BufferedReader(new FileReader(new File(DATA_SWAP)));
            answer = br.readLine();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return answer;
    }

    public static void execPy() {
        Process proc = null;
        try {
            //proc = Runtime.getRuntime().exec("python3 " + PY_URL + " --path images2/11.jpg");
            proc = Runtime.getRuntime().exec("python3 motor_detection.py --path images2/11.jpg");
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    // 测试码
    public static void main(String[] args) throws IOException, InterruptedException {
        
        writeImagePath();
        execPy();
        System.out.println("output:");
        System.out.println(readAnswer());
    }
}